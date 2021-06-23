from mmdet.datasets import DATASETS, XMLDataset
import xml.etree.ElementTree as ET
import mmcv
import os.path as osp
from PIL import Image
import numpy as np
from ..core.evaluation import eval_mapR_flexible


@DATASETS.register_module()
class UAVBD(XMLDataset):
    CLASSES = ('bottle',)

    def __init__(self, difficult_threshold=1000, **kwargs):
        self.difficult_thres = difficult_threshold
        super(UAVBD, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = osp.join('images', f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, 'labels', f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, 'labels', f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'labels', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('robndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                float(bnd_box.find('cx').text),
                float(bnd_box.find('cy').text),
                float(bnd_box.find('w').text),
                float(bnd_box.find('h').text),
                float(bnd_box.find('angle').text),
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2]
                h = bbox[3]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult > self.difficult_thres or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 5))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 5))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'labels', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids

    def get_ann_info_test(self, idx):
        ann_info = self.get_ann_info(idx)

        gt_bboxes = np.concatenate(
            (ann_info['bboxes'], ann_info['bboxes_ignore']), axis=0)
        gt_labels = np.concatenate(
            (ann_info['labels'], ann_info['labels_ignore']), axis=0)
        ignore = np.concatenate((
            np.zeros(len(ann_info['bboxes']), dtype=np.bool),
            np.ones(len(ann_info['bboxes_ignore']), dtype=np.bool)), axis=0)
        gt_attrs = dict(ignore=ignore)

        ann = dict(
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_attrs=gt_attrs)

        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['cowa']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info_test(i) for i in range(len(self))]
        if metric == 'cowa':
            eval_results = eval_mapR_flexible(
                results,
                annotations,
                iou_thrs=[0.5 + 0.05 * x for x in range(10)],
                breakdown=[
                    dict(
                        type='ScaleBreakdown',
                        scale_ranges=dict(
                            Scale_S=(0, 32),
                            Scale_L=(32, 10000)))
                ],
                report_config=[
                    ('map', lambda x: x['breakdown'] == 'All'),
                    ('map50', lambda x: x['iou_threshold'] == 0.5 and x[
                        'breakdown'] == 'All'),
                    ('map75', lambda x: x['iou_threshold'] == 0.75 and x[
                        'breakdown'] == 'All'),
                    ('s_map', lambda x: x['breakdown'] == 'Scale_S'),
                    ('l_map', lambda x: x['breakdown'] == 'Scale_L')
                ],
                classes=self.CLASSES,
                iou_calculator=dict(type='IOUR'),
                matcher=dict(type='MatcherCoCo'),
                nproc=0,
                logger=logger)
        return eval_results
