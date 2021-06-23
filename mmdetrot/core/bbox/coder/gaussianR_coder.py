import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class GaussianRBBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps

    @classmethod
    def representation(cls, bboxes):
        _shape = bboxes.shape
        xywhr = bboxes.reshape(-1, 5)
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1)
        R = R.reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        Sigma = R.matmul(S.square()).matmul(R.permute(0, 2, 1))
        xy_dev = Sigma.diagonal(dim1=-2, dim2=-1).clamp(0).sqrt()
        xy_r = Sigma[..., 0, 1] / xy_dev.prod(dim=-1)
        rep = torch.cat((xy, xy_dev, xy_r.unsqueeze(-1)), dim=-1).reshape(
            _shape[:-1] + (5,))
        return rep

    @classmethod
    def explanation(cls, rep):
        rep_shape = rep.shape
        rep = rep.reshape(-1, 5)
        xy = rep[:, :2]
        xy_stddev = rep[:, 2:4]
        xy_covar = rep[:, 4] * xy_stddev.prod(dim=-1)
        xy_var = xy_stddev.square()

        b = - xy_var.sum(dim=-1)
        judge = ((xy_var[:, 0] - xy_var[:,
                                 1]).square() + 4 * xy_covar.square()).sqrt()
        h2 = (-b + judge) / 2
        w2 = (-b - judge) / 2

        r = torch.atan2(w2 - xy_var[:, 0], xy_covar)
        r[xy_covar == 0] = 0

        h = 2 * h2.clamp(min=1e-7, max=1e7).sqrt()
        w = 2 * w2.clamp(min=1e-7, max=1e7).sqrt()

        decoded_bboxes = torch.cat((xy, torch.stack((w, h, r), dim=-1)),
                                   dim=-1)
        decoded_bboxes = decoded_bboxes.reshape(rep_shape[:-1] + (5,))
        return decoded_bboxes

    def encode(self, bboxes, gt_bboxes):
        raise NotImplementedError

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 5
        pred_bboxes = pred_bboxes.sigmoid()
        pred_offset = stride * (pred_bboxes[..., :2] * 2. - 1.)
        pred_stddev = (pred_bboxes[..., 2:4] * 2.) ** 2.
        pred_pearson = pred_bboxes[..., 4] * 2. - 1.

        # Get outputs x, y
        xy = bboxes[..., :2] + pred_offset
        xy_stddev = bboxes[..., 2:4] * pred_stddev

        decoded_bboxes = torch.cat((xy, xy_stddev, pred_pearson.unsqueeze(-1)),
                                   dim=-1)

        return decoded_bboxes
