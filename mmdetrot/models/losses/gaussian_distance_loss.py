import torch
from torch import nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
from copy import deepcopy


def xywhr2xyrs(xywhr):
    xywhr = xywhr.reshape(-1, 5)
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S


def xyzwhlr2xyzrsl(xyzwhlr):
    xyzwhlr = xyzwhlr.reshape(-1, 7)
    xyz = xyzwhlr[..., :3]
    wh = xyzwhlr[..., 3:5].clamp(min=1e-7, max=1e7)
    l = xyzwhlr[..., 5].clamp(min=1e-7, max=1e7)
    r = xyzwhlr[..., 6]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    Sl = 0.5 * l
    return xyz, R, S, Sl


def postprocess(distance, fun='log1p', tau=1.0):
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


@weighted_loss
def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z

    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2

    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)

    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))

    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))

    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)

    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z)))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))

    whr_distance = S_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whr_distance = whr_distance + S_t.diagonal(dim1=-2, dim2=-1).square().sum(
        dim=-1)
    _t = Sigma_p.matmul(Sigma_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = S_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * S_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        wh_p = pred[..., 2:4].clamp(min=1e-7, max=1e7)
        wh_t = target[..., 2:4].clamp(min=1e-7, max=1e7)
        scale = ((wh_p.log() + wh_t.log()).sum(dim=-1) / 4).exp()
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def gwd3d_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """
    pred and target are modeled as 3-multivariate normal distribution
    because no pitch and no roll is considered,
    the conv matrices can be expressed in the form:

        |Σwh  O|
    Σ = |O   Σl|, where Σl is a scalar

    in this epecial case we have:

      |Σwh  O|
    Tr|O   Σl| = TrΣwh + Σl                 -----------------1

    |Σwh  O|^(1/2)   |Σwh^(1/2)  O|
    |O   Σl|       = |O   sqrt(Σl)|         -----------------2

    |Σwh1  O|   |Σwh2  O|   |Σwh1*Σwh2  O|
    |O   Σl1| * |O   Σl2| = |O    Σl1*Σl2|  -----------------3

    formula 1 gives:
    TrΣp = TrΣwhp + Σlp
    TrΣt = TrΣwht + Σlt

    combination of formula 1, 2 and 3 gives:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr((Σwhp^(1/2) * Σwht * Σwhp^(1/2))^(1/2)) + Σlp^(1/2) * Σlt^(1/2)

    gwd3d of pred and target can thus be expressed as:
    gwd3d^2(P, T)
    = gwd^2(Pwh, Twh) + (zp - zt)^2 + Σlp + Σlt - 2 * Σlp^(1/2) * Σlt^(1/2)
    = gwd^2(Pwh, Twh) + (zp - zt)^2 + (Σlp^(1/2) - Σlt^(1/2))^2
    """
    xyz_p, Rwh_p, Swh_p, Sl_p = xyzwhlr2xyzrsl(pred)
    xyz_t, Rwh_t, Swh_t, Sl_t = xyzwhlr2xyzrsl(target)

    xyz_distance = (xyz_p - xyz_t).square().sum(dim=-1)

    whlr_distance = Swh_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whlr_distance = whlr_distance + Swh_t.diagonal(dim1=-2,
                                                   dim2=-1).square().sum(
        dim=-1)

    Sigmawh_p = Rwh_p.matmul(Swh_p.square()).matmul(Rwh_p.permute(0, 2, 1))
    Sigmawh_t = Rwh_t.matmul(Swh_t.square()).matmul(Rwh_t.permute(0, 2, 1))
    _t = Sigmawh_p.matmul(Sigmawh_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = Swh_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * Swh_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)

    whlr_distance = whlr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    whlr_distance = whlr_distance + (Sl_p - Sl_t).square()

    distance = (xyz_distance + alpha * alpha * whlr_distance).clamp(0).sqrt()

    if normalize:
        whl_p = pred[..., 3:6].clamp(min=1e-7, max=1e7)
        whl_t = target[..., 3:6].clamp(min=1e-7, max=1e7)
        scale = ((whl_p.log() + whl_t.log()).sum(dim=-1) / 6).exp()
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)
    S_p_inv = S_p.diagonal(dim1=-2, dim2=-1).reciprocal().diag_embed()
    Sigma_p_inv = R_p.matmul(S_p_inv.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).matmul(Sigma_p_inv).matmul(
        dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.matmul(
        Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_sqrt_log = S_p.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    Sigma_t_det_sqrt_log = S_t.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    whr_distance = whr_distance + (Sigma_p_det_sqrt_log - Sigma_t_det_sqrt_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def kld3d_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    xyz_p, Rwh_p, Swh_p, Sl_p = xyzwhlr2xyzrsl(pred)
    xyz_t, Rwh_t, Swh_t, Sl_t = xyzwhlr2xyzrsl(target)

    Swh_p_inv = Swh_p.diagonal(dim1=-2, dim2=-1).reciprocal().diag_embed()
    Sl_p_inv = Sl_p.reciprocal()
    Sigmawh_p_inv = Rwh_p.matmul(Swh_p_inv.square()).matmul(
        Rwh_p.permute(0, 2, 1))
    Sigmawh_t = Rwh_t.matmul(Swh_t.square()).matmul(Rwh_t.permute(0, 2, 1))

    dxy = (xyz_p[..., :2] - xyz_t[..., :2]).unsqueeze(-1)
    dz = xyz_p[..., 2] - xyz_t[..., 2]
    xyz_distance = 0.5 * dxy.permute(0, 2, 1).matmul(Sigmawh_p_inv).matmul(
        dxy).view(-1)
    xyz_distance = xyz_distance + 0.5 * dz.square() * Sl_p_inv.square()

    whlr_distance = 0.5 * Sigmawh_p_inv.matmul(
        Sigmawh_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whlr_distance = whlr_distance + 0.5 * Sl_p_inv.square() * Sl_t.square()

    Sigma_p_det_sqrt_log = Swh_p.diagonal(dim1=-2, dim2=-1).log().sum(
        dim=-1) + Sl_p.log()
    Sigma_t_det_sqrt_log = Swh_t.diagonal(dim1=-2, dim2=-1).log().sum(
        dim=-1) + Sl_t.log()
    whlr_distance = whlr_distance + (
            Sigma_p_det_sqrt_log - Sigma_t_det_sqrt_log)
    whlr_distance = whlr_distance - 1.5
    distance = (xyz_distance / (alpha * alpha) + whlr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def jd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    jd = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=False,
                  reduction='none')
    jd = jd + kld_loss(target, pred, fun='none', tau=0, alpha=alpha,
                       sqrt=False,
                       reduction='none')
    jd = jd * 0.5
    if sqrt:
        jd = jd.clamp(0).sqrt()
    return postprocess(jd, fun=fun, tau=tau)


@weighted_loss
def jd3d_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    jd = kld3d_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=False,
                    reduction='none')
    jd = jd + kld3d_loss(target, pred, fun='none', tau=0, alpha=alpha,
                         sqrt=False, reduction='none')
    jd = jd * 0.5
    if sqrt:
        jd = jd.clamp(0).sqrt()
    return postprocess(jd, fun=fun, tau=tau)


@weighted_loss
def kld_symmax_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    kld_pt = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_tp = kld_loss(target, pred, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_symmax = torch.max(kld_pt, kld_tp)
    return postprocess(kld_symmax, fun=fun, tau=tau)


@weighted_loss
def kld3d_symmax_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0,
                      sqrt=True):
    kld_pt = kld3d_loss(pred, target, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_tp = kld3d_loss(target, pred, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_symmax = torch.max(kld_pt, kld_tp)
    return postprocess(kld_symmax, fun=fun, tau=tau)


@weighted_loss
def kld_symmin_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    kld_pt = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_tp = kld_loss(target, pred, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_symmin = torch.min(kld_pt, kld_tp)
    return postprocess(kld_symmin, fun=fun, tau=tau)


@weighted_loss
def kld3d_symmin_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0,
                      sqrt=True):
    kld_pt = kld3d_loss(pred, target, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_tp = kld3d_loss(target, pred, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_symmin = torch.min(kld_pt, kld_tp)
    return postprocess(kld_symmin, fun=fun, tau=tau)


@LOSSES.register_module()
class GDLoss(nn.Module):
    BAG_GD_LOSS = {'gwd': gwd_loss,
                   'gwd3d': gwd3d_loss,
                   'kld': kld_loss,
                   'kld3d': kld3d_loss,
                   'jd': jd_loss,
                   'jd3d': jd3d_loss,
                   'kld_symmax': kld_symmax_loss,
                   'kld3d_symmax': kld3d_symmax_loss,
                   'kld_symmin': kld_symmin_loss,
                   'kld3d_symmin': kld3d_symmin_loss}

    def __init__(self, loss_type, fun='log1p', tau=1.0, alpha=1.0,
                 reduction='mean', loss_weight=1.0, **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

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
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)
        return self.loss(
            pred,
            target,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **_kwargs) * self.loss_weight
