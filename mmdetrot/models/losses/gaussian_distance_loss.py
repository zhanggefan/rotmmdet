import torch
from torch import nn
from ...ops.gaussian_distance_loss import matsqrt2x2sym
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


def xywhr2xyrs(xywhr):
    xywhr = xywhr.reshape(-1, 5)
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4]
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S


@weighted_loss
def gwd_loss(pred, target, fun='log', tau=1.0, alpha=1):
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)

    _t = (xy_p - xy_t)
    xy_distance = (_t * _t).sum(dim=-1)

    Sigma_p_sqrt = R_p.matmul(S_p).matmul(R_p.permute(0, 2, 1))
    Sigma_p = R_p.matmul(S_p * S_p).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t * S_t).matmul(R_t.permute(0, 2, 1))

    whr_distance = Sigma_p[..., 0, 0] + Sigma_p[..., 1, 1]
    whr_distance += Sigma_t[..., 0, 0] + Sigma_t[..., 1, 1]

    _t = Sigma_p_sqrt.matmul(Sigma_t).matmul(Sigma_p_sqrt)
    _t = matsqrt2x2sym(_t.flatten(start_dim=-2)[..., [0, 1, 3]])
    whr_distance += (-2) * (_t[..., 0] + _t[..., 2])

    distance = (xy_distance + alpha * whr_distance).clamp_(min=0).sqrt()

    scale = (pred[..., 2:4] * target[..., 2:4]).prod(dim=-1).sqrt().sqrt()
    distance = distance / scale

    if fun == 'log':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        # for numerical stability
        distance = torch.sqrt(distance)
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun} for gwd loss')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


@LOSSES.register_module()
class GWDLoss(nn.Module):
    def __init__(self, fun='log', tau=1.0, alpha=1.0, reduction='mean',
                 loss_weight=1.0):
        super(GWDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log', 'sqrt', 'none']
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        return gwd_loss(
            pred,
            target,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction) * self.loss_weight
