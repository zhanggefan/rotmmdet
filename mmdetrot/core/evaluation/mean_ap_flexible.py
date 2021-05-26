from collections import OrderedDict
from os import cpu_count

import numpy as np
from mmcv.utils import Registry, build_from_cfg
from mmcv.utils.progressbar import track_iter_progress, track_parallel_progress
from mmdet.core.evaluation.mean_ap import average_precision

from ...ops.eval_utils.iou import iou_coco
from ...ops.eval_utils.match import match_coco

EVAL_BREAKDOWN = Registry('Evaluation Breakdown')

EVAL_IOU_CALCULATOR = Registry('Evaluation IOU calculator')

EVAL_MATCHER = Registry('Evaluation Matcher')


@EVAL_IOU_CALCULATOR.register_module()
class IOU2DCoCo:

    def __call__(self, det_bboxes, gt_bboxes, gt_iscrowd=None):
        if gt_iscrowd is None:
            gt_iscrowd = gt_bboxes.new_zeros(gt_bboxes.shape[0], dtype=np.bool)
        return iou_coco(det_bboxes, gt_bboxes, gt_iscrowd)


@EVAL_MATCHER.register_module()
class MatcherCoCo:

    def __call__(self, ious, iou_thrs, gt_isignore=None, gt_iscrowd=None):
        if gt_iscrowd is None:
            gt_iscrowd = ious.new_zeros(ious.shape[1], dtype=np.bool)
        if gt_isignore is None:
            gt_isignore = ious.new_zeros(ious.shape[1], dtype=np.bool)
        return match_coco(ious, iou_thrs, gt_isignore, gt_iscrowd)


class NoBreakdown:

    def __init__(self, classes, apply_to=None, *args, **kwargs):
        if apply_to is None:
            apply_to = classes
        self.classes = classes
        self.apply_to = apply_to
        self.names = ['All']

    def breakdown_flags(self, boxes, attrs=None):
        num_boxes = len(boxes)
        flags = np.ones((1, num_boxes), dtype=np.bool)
        if attrs is not None and 'ignore' in attrs:
            flags[:, attrs['ignore']] = False
        return flags

    def breakdown(self, boxes, label, attrs=None):
        flags = self.breakdown_flags(boxes, attrs)
        if self.classes[label] in self.apply_to:
            return flags
        else:
            return flags[:0]

    def breakdown_names(self, label):
        if self.classes[label] in self.apply_to:
            return [f'{n}' for n in self.names]
        else:
            return []


@EVAL_BREAKDOWN.register_module()
class ScaleBreakdown(NoBreakdown):

    def __init__(self, scale_ranges, classes, apply_to=None, *args, **kwargs):
        super(ScaleBreakdown, self).__init__(classes, apply_to, *args,
                                             **kwargs)
        self.names = []
        self.area_ranges = []
        for k in scale_ranges:
            self.names.append(k)
            smin, smax = scale_ranges[k]
            self.area_ranges.append((smin * smin, smax * smax))

    def breakdown_flags(self, boxes, attrs=None):
        num_ranges = len(self.area_ranges)
        num_boxes = len(boxes)
        if attrs is not None and 'area' in attrs:
            area = attrs['area']
        else:
            wh = boxes[:, 2:] - boxes[:, :2]
            area = wh[:, 0] * wh[:, 1]
        area_flags = np.zeros((num_ranges, num_boxes), dtype=np.bool)
        for dist_idx, (min_area, max_area) in enumerate(self.area_ranges):
            area_flags[dist_idx][(area >= min_area) & (area < max_area)] = True
        if attrs is not None and 'ignore' in attrs:
            area_flags[:, attrs['ignore']] = False
        return area_flags


class FlexibleStatisticsEval(object):

    def __init__(self, classes, iou_thrs, breakdown, iou_calculator, matcher,
                 nproc):
        self.classes = classes
        self.iou_thrs = iou_thrs
        self.breakdown = [NoBreakdown(classes)]
        self.breakdown += [
            build_from_cfg(
                bkd, EVAL_BREAKDOWN, default_args=dict(classes=classes))
            for bkd in breakdown
        ]
        self.iou_calculator = build_from_cfg(iou_calculator,
                                             EVAL_IOU_CALCULATOR)
        self.matcher = build_from_cfg(matcher, EVAL_MATCHER)
        self.nproc = nproc

    def statistics_single(self, input):
        """Check if detected bboxes are true positive or false positive."""
        tp_score_info = []
        det, anno = input
        num_cls = len(det)
        num_iou_thrs = len(self.iou_thrs)

        gt_bboxes = anno['gt_bboxes']
        gt_labels = anno['gt_labels']
        gt_attrs = anno['gt_attrs']

        for cls in range(num_cls):
            # prepare detections
            cls_name = self.classes[cls] if self.classes is not None else cls

            cls_det_bboxes = det[cls][:, :-1]
            cls_det_scores = det[cls][:, -1]
            sort_ind = cls_det_scores.argsort()[::-1]
            cls_det_bboxes = cls_det_bboxes[sort_ind]
            cls_det_scores = cls_det_scores[sort_ind]
            cls_num_dets = cls_det_scores.shape[0]

            # prepare ground-truths
            cls_gt_msk = gt_labels == cls
            cls_gt_bboxes = gt_bboxes[cls_gt_msk]
            cls_gt_attrs = {k: v[cls_gt_msk] for k, v in gt_attrs.items()}
            if 'ignore' in cls_gt_attrs:
                cls_gt_ignore_msk = cls_gt_attrs['ignore']
                cls_num_ignore_gts = np.count_nonzero(cls_gt_ignore_msk)
            else:
                cls_gt_ignore_msk = np.zeros(len(cls_gt_bboxes), dtype=np.bool)
                cls_num_ignore_gts = 0
            if 'iscrowd' in cls_gt_attrs:
                cls_gt_crowd_msk = cls_gt_attrs['iscrowd']
            else:
                cls_gt_crowd_msk = np.zeros(len(cls_gt_bboxes), dtype=np.bool)

            cls_num_gts = len(cls_gt_bboxes) - cls_num_ignore_gts

            # prepare breakdown masks
            cls_det_bkd = []
            cls_gt_bkd = []
            cls_bkd_names = []
            for fun in self.breakdown:
                cls_det_bkd.append(fun.breakdown(cls_det_bboxes, cls))
                cls_gt_bkd.append(
                    fun.breakdown(cls_gt_bboxes, cls, cls_gt_attrs))
                cls_bkd_names += fun.breakdown_names(cls)
            cls_det_bkd = np.concatenate(cls_det_bkd, axis=0)
            cls_gt_bkd = np.concatenate(cls_gt_bkd, axis=0)
            num_bkd = cls_gt_bkd.shape[0]

            # all detections are false positive by default
            cls_tp = np.zeros((num_iou_thrs, cls_num_dets), dtype=np.bool)

            # calculate num gt (not considering ignored gt boxes)
            cls_gt_count = []
            for bkd_idx in range(num_bkd):
                cls_gt_count.append(np.count_nonzero(cls_gt_bkd[bkd_idx]))

            # handling empty det or empty gt
            if (cls_num_gts + cls_num_ignore_gts) == 0 or cls_num_dets == 0:
                for bkd_idx in range(num_bkd):
                    tp_score_info.append(
                        (cls_name, cls_bkd_names[bkd_idx],
                         cls_gt_count[bkd_idx], cls_det_scores, cls_tp,
                         cls_det_bkd[bkd_idx:bkd_idx + 1].repeat(
                             num_iou_thrs, axis=0)))
            else:
                ious = self.iou_calculator(cls_det_bboxes, cls_gt_bboxes,
                                           cls_gt_crowd_msk)

                for bkd_idx in range(num_bkd):
                    cls_gt_bkd_msk = cls_gt_bkd[bkd_idx]
                    matched_gt_idx = self.matcher(
                        ious, np.array(self.iou_thrs, dtype=np.float32),
                        (~cls_gt_bkd_msk), cls_gt_crowd_msk.astype(np.bool))

                    cls_tp[...] = False
                    cls_tp[matched_gt_idx > -1] = True

                    _msk_fp = (
                        cls_det_bkd[bkd_idx:bkd_idx + 1] &
                        (matched_gt_idx == -1))
                    _msk_tp = ((cls_gt_bkd_msk[matched_gt_idx]) &
                               (matched_gt_idx > -1))
                    _msk_fptp = (_msk_fp | _msk_tp)
                    tp_score_info.append((cls_name, cls_bkd_names[bkd_idx],
                                          cls_gt_count[bkd_idx],
                                          cls_det_scores, cls_tp, _msk_fptp))

        return tp_score_info

    def statistics_accumulate(self, input):
        cls, bkd, num_gt, score, tp, bkd_msk = input
        eval_result_list = []
        rank = score.argsort()[::-1]
        tp = tp[:, rank]
        bkd_msk = bkd_msk[:, rank]
        for iou_thr_idx, iou_thr in enumerate(self.iou_thrs):
            tpcumsum = tp[iou_thr_idx, bkd_msk[iou_thr_idx]].cumsum()
            num_det = len(tpcumsum)
            recall = tpcumsum / max(num_gt, 1e-7)
            precision = tpcumsum / np.arange(1, num_det + 1)
            m_ap = average_precision(recall, precision)
            max_recall = recall.max() if len(recall) > 0 else 0
            key = dict(class_name=cls, breakdown=bkd, iou_threshold=iou_thr)
            value = dict(
                num_det=num_det, num_gt=num_gt, recall=max_recall, mAP=m_ap)
            eval_result_list.append((key, value))
        return eval_result_list

    def statistics_eval(self, det_results, annotations):
        if self.nproc == 0:
            tp_score_infos = [
                self.statistics_single(d)
                for d in zip(track_iter_progress(det_results), annotations)
            ]
        else:
            tp_score_infos = track_parallel_progress(
                func=self.statistics_single,
                tasks=[*zip(det_results, annotations)],
                nproc=self.nproc,
                chunksize=16)

        tp_score_infos_all = []
        for cls_bkd_item in zip(*tp_score_infos):
            (cls, bkd, num_gt, score, tp, bkd_msk) = tuple(zip(*cls_bkd_item))
            assert len(set(cls)) == 1
            assert len(set(bkd)) == 1
            num_gt = sum(num_gt)
            score = np.concatenate(score, axis=0)
            tp = np.concatenate(tp, axis=1)
            bkd_msk = np.concatenate(bkd_msk, axis=1)
            tp_score_infos_all.append(
                (cls[0], bkd[0], num_gt, score, tp, bkd_msk))

        if self.nproc == 0:
            eval_result_list = [
                self.statistics_accumulate(d)
                for d in track_iter_progress(tp_score_infos_all)
            ]
        else:
            eval_result_list = track_parallel_progress(
                func=self.statistics_accumulate,
                tasks=tp_score_infos_all,
                nproc=self.nproc,
                chunksize=16)

        eval_result_list = sum(eval_result_list, [])
        return eval_result_list

    def report(self, eval_result_list, group_by):
        report_dict = OrderedDict()
        for name, cond in group_by:
            cond_met_map = []
            for k, v in eval_result_list:
                if cond(k) and v['num_gt'] > 0:
                    cond_met_map.append(v['mAP'])
            cond_met_map = np.mean(cond_met_map)
            report_dict[name] = cond_met_map
        return report_dict


def eval_map_flexible(det_results,
                      annotations,
                      iou_thrs=[0.5],
                      breakdown=[],
                      iou_calculator=dict(type='IOU2DCoCo'),
                      matcher=dict(type='MatcherCoCo'),
                      classes=None,
                      logger=None,
                      report_config=[('map', lambda x: x['breakdown'] == 'All')
                                     ],
                      nproc=None):
    assert len(det_results) == len(annotations)

    if nproc is None:
        nproc = 0
    elif nproc < 0:
        nproc = cpu_count() or 0

    fse = FlexibleStatisticsEval(classes, iou_thrs, breakdown, iou_calculator,
                                 matcher, nproc)

    eval_result_list = fse.statistics_eval(det_results, annotations)
    report = fse.report(eval_result_list, report_config)
    return report
