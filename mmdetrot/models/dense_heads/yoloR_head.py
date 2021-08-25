from .yolocsp_head import YOLOCSPHead
import torch
from mmcv.cnn import normal_init
import math
from mmdet.models.builder import HEADS
from mmdet.models.losses import reduce_loss
from ...core.post_processing import multiclass_nmsR


@HEADS.register_module()
class YOLORHead(YOLOCSPHead):

    def __init__(self,
                 anchor_generator=dict(
                     type='YOLOGaussianRAnchorGenerator',
                     base_sizes=[
                         [(10, 20), (20, 10), (14.1, 14.1)],  # P3/8
                         [(20, 40), (40, 20), (28.3, 28.3)],  # P4/16
                         [(40, 80), (80, 40), (56.6, 56.6)]
                     ],  # P5/32
                     strides=[8, 16, 32]),
                 bbox_coder=dict(type='GaussianRBBoxCoder'),
                 loss_bbox=dict(type='GDLoss', loss_type='kld',
                                loss_weight=3.2),
                 **kwargs):
        super(YOLORHead, self).__init__(anchor_generator=anchor_generator,
                                        bbox_coder=bbox_coder,
                                        loss_bbox=loss_bbox,
                                        **kwargs)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (5) +
        objectness (1) + num_classes"""

        if not self.class_agnostic:
            return 6 + self.num_classes
        else:
            return 6

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            normal_init(m, std=0.01)
        for m, stride in zip(self.convs_pred, self.featmap_strides):
            b = m.bias.view(-1, self.num_attrib)  # conv.bias(255) to (3,85)
            b[:, 5] += math.log(
                self.num_obj_avg /
                (480 / stride) ** 2)  # obj (8 objects per 640 image)
            if not self.class_agnostic:
                b[:, 6:] += math.log(
                    0.6 / (self.num_classes -
                           0.99)) if self.class_freq is None else torch.log(
                    self.class_freq / self.class_freq.sum())  # cls
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def loss_single_no_assigner(self, pred_map, anchors, stride, pos_indices,
                                target_bboxes, target_labels):
        """Compute loss of a single level.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            anchors (Tensor): anchors for a single level.
                has shape (num_anchors, 4)
            pos_indices (tuple[Tensor]): positive sample indices.
                (img_idx, anchor_idx), each indices tensor has shape
                (k,), which stands for k positive samples
            pos_bboxes (Tensor): target tensor for positive samples.
                has shape (k, 4)
            pos_labels (Tensor): target tensor for positive samples.
                has shape (k, self.num_classes)

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)

        img_ind, anchor_ind = pos_indices

        pred_conf = pred_map[..., 5]
        target_conf = torch.zeros_like(pred_conf, requires_grad=False)

        loss_bbox = pred_map.new_zeros((1,))
        loss_cls = pred_map.new_zeros((1,))

        if anchor_ind.numel():
            pred_map_pos = pred_map[img_ind, anchor_ind]
            anchor_pos = anchors[anchor_ind]

            # apply transforms on bbox prediction
            pred_bbox = pred_map_pos[..., :5]
            pred_bbox = self.bbox_coder.decode(anchor_pos, pred_bbox, stride)

            target_bboxes = self.bbox_coder.representation(target_bboxes)

            gd_loss = self.loss_bbox(
                pred_bbox, target_bboxes, reduction_override='none')

            loss_bbox += reduce_loss(
                gd_loss, reduction=self.loss_bbox.reduction)

            pred_cls = pred_map_pos[..., 6:]
            target_cls = target_labels

            if not self.class_agnostic:
                loss_cls += self.loss_cls(pred_cls, target_cls)

            _r = self.conf_iou_loss_ratio
            _conf_t = (1 - _r) + _r * (1 - gd_loss).detach().clamp(0.0, 1.0)
            target_conf[pos_indices] = _conf_t.type(target_conf.dtype)

        loss_conf = self.loss_conf(pred_conf, target_conf)

        return loss_cls, loss_conf, loss_bbox * self.loss_bbox_weight

    # @force_fp32(apply_to=('pred_maps',))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        num_levels = len(pred_maps)
        num_image = len(img_metas)

        featmap_sizes = [pred_maps[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, pred_maps[0].device)

        mlvl_bbox_pred = []
        mlvl_conf_pred = []
        mlvl_score_pred = []

        for lvl in range(num_levels):
            lvl_pred_maps = pred_maps[lvl].permute(0, 2, 3, 1).reshape(
                (num_image, -1, self.num_attrib))
            # class score
            if not self.class_agnostic:
                mlvl_score_pred.append(lvl_pred_maps[:, :, 6:])
            # conf score
            mlvl_conf_pred.append(lvl_pred_maps[:, :, 5].sigmoid())
            # bbox transform
            lvl_bbox_pred = lvl_pred_maps[:, :, :5].reshape(-1, 5)
            lvl_anchors = mlvl_anchors[lvl][None, ...].repeat(
                (num_image, 1, 1))

            lvl_bbox_pred = self.bbox_coder.decode(
                bboxes=lvl_anchors.reshape(-1, 5),
                pred_bboxes=lvl_bbox_pred.reshape(-1, 5),
                stride=self.featmap_strides[lvl])
            lvl_bbox_pred = self.bbox_coder.explanation(lvl_bbox_pred)
            lvl_bbox_pred = lvl_bbox_pred.reshape((num_image, -1, 5))
            mlvl_bbox_pred.append(lvl_bbox_pred)

        if not self.class_agnostic:
            mimg_score_pred = [
                score for score in torch.cat(mlvl_score_pred, dim=1)
            ]
        else:
            mimg_score_pred = None
        mimg_conf_pred = [conf for conf in torch.cat(mlvl_conf_pred, dim=1)]
        mimg_bbox_pred = [bbox for bbox in torch.cat(mlvl_bbox_pred, dim=1)]

        result_list = []

        for img_id in range(len(img_metas)):
            scale_factor = img_metas[img_id]['scale_factor']
            assert scale_factor.min() == scale_factor.max()
            scale_factor = scale_factor.min()
            proposals = self._get_bboxes_single(
                cls_pred=mimg_score_pred[img_id]
                if mimg_score_pred is not None else None,
                conf_pred=mimg_conf_pred[img_id],
                bbox_pred=mimg_bbox_pred[img_id],
                scale_factor=scale_factor,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_pred,
                           conf_pred,
                           bbox_pred,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_pred (Tensor): Score of each predicted bbox of a single image.
            conf_pred (Tensor): Confidence of each predicted bbox of a single
                image.
            bbox_pred (Tensor): Predicted bbox of a single image.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg

        # Get top-k prediction
        nms_pre = cfg.get('nms_pre', -1)

        if 0 < nms_pre < conf_pred.size(0):
            _, topk_inds = conf_pred.topk(nms_pre)
            bbox_pred = bbox_pred[topk_inds, :]
            if not self.class_agnostic:
                cls_pred = cls_pred[topk_inds, :]
            conf_pred = conf_pred[topk_inds]

        if not self.class_agnostic:
            cls_pred *= conf_pred[:, None]
        else:
            cls_pred = conf_pred[:, None]

        if with_nms and (cls_pred.size(0) == 0):
            return torch.zeros((0, 6)), torch.zeros((0,))

        if rescale:
            bbox_pred[..., :4] /= bbox_pred.new_tensor(scale_factor)

        if with_nms:
            # In mmdet 2.x, the class_id for background is num_classes.
            # i.e., the last column.
            padding = cls_pred.new_zeros(cls_pred.shape[0], 1)
            cls_pred = torch.cat([cls_pred, padding], dim=1)

            det_bboxes, det_labels = multiclass_nmsR(bbox_pred, cls_pred,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            cls_pred = cls_pred * conf_pred[:, None]
            class_score, class_id = cls_pred.max(dim=-1)
            return torch.cat((bbox_pred, class_score[:, None]),
                             dim=-1), class_id
