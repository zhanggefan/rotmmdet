import torch

from mmdet.core.anchor.builder import ANCHOR_GENERATORS
from mmdet.core.anchor.anchor_generator import YOLOAnchorGenerator
from torch.nn.modules.utils import _pair
from ..bbox.coder.gaussianR_coder import GaussianRBBoxCoder


@ANCHOR_GENERATORS.register_module()
class YOLOGaussianRAnchorGenerator(YOLOAnchorGenerator):
    """Anchor generator for YOLOR.
    """

    def __init__(self, strides, base_sizes):
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(base_sizes_per_level)
        self.base_anchors = self.gen_base_anchors()

    def gen_single_level_base_anchors(self, base_sizes_per_level, center=None):
        """Generate base anchors of a single level.

        Args:
            base_sizes_per_level (list[tuple[int, int]]): Basic sizes of
                anchors.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            x_stddev, y_stddev = base_size

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = torch.Tensor([
                x_center, y_center, x_stddev, y_stddev, 0
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors

    def responsible_indices(self, featmap_sizes, gt_bboxes_list, neighbor=3,
                            shape_match_thres=4., device='cuda'):
        """Generate responsible anchor flags of grid cells in multiple scales.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in multiple
                feature levels.
            gt_bboxes_list (list(Tensor)): List of Ground truth boxes, each with shape (n, 5). encoded
            neighbor (int): assign gt to neighbor grid cell. Possible values:
                0: assign prediction responsibility to the only one grid cell where the center of the gt bbox locates
                2: additionally assign prediction responsibility to 2 nearest neighbor grid cells, like what yolo v5 do
                3: additionally assign prediction responsibility to all 3 neighbor grid cells
            shape_match_thres (float): shape matching threshold between base_anchors and gt-bboxes
                matched gt-bboxes and base_anchors shall meet the following requirements:
                    1.0 / shape_match_thres < (height(gt-bboxes) / height(base_anchors)) < shape_match_thres
                    1.0 / shape_match_thres < (width(gt-bboxes) / width(base_anchors)) < shape_match_thres
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Return:
            list(tuple(torch.Tensor)): responsible indices
        """
        # Build targets for compute_loss(), input targets(x,y,w,h)
        img_id = []

        for ind, gt_bboxes in enumerate(gt_bboxes_list):
            num_gt = gt_bboxes.shape[0]
            img_id.append(gt_bboxes.new_full((num_gt,), ind, dtype=torch.long))

        gt_bboxes = torch.cat(gt_bboxes_list, dim=0)
        gt_bboxes = GaussianRBBoxCoder.representation(gt_bboxes)
        img_id = torch.cat(img_id, dim=0)

        indices = []

        if gt_bboxes.shape[0] == 0:
            for _ in range(self.num_levels):
                indices.append(
                    (torch.tensor([], device=device, dtype=torch.long),
                     torch.tensor([], device=device, dtype=torch.long),
                     torch.tensor([], device=device, dtype=torch.long)))
            return indices
        gt_xy = gt_bboxes[:, :2].to(device)
        gt_stddev = gt_bboxes[:, 2:4].to(device)

        neighbor_offset = gt_xy.new_tensor([[0, 0],  # current grid
                                            [-1, 0],  # left neighbor grid
                                            [0, -1],  # upper neighbor grid
                                            [1, 0],  # right neighbor grid
                                            [0, 1],  # lower neighbor grid
                                            [-1, -1],
                                            # upper-left neighbor grid
                                            [1, -1],
                                            # upper-right neighbor grid
                                            [1, 1],
                                            # lower-right neighbor grid
                                            [-1,
                                             1]])  # lower-left neighbor grid

        for i in range(self.num_levels):
            feat_h, feat_w = featmap_sizes[i]
            strides = self.strides[i]
            num_base_anchors = self.num_base_anchors[i]

            base_anchors = self.base_anchors[i].to(device)
            base_anchor_stddev = base_anchors[:, 2:4]

            # perform shape matching between anchors and gt-boxes
            # the shape of result tensor shape_match: (num_anchors, num_gt)
            extent_deviation = (
                    gt_stddev[None, :, :] / base_anchor_stddev[:, None, :])
            extent_deviation = torch.max(extent_deviation,
                                         1. / extent_deviation).max(
                dim=2).values
            shape_match = extent_deviation < shape_match_thres
            base_anchor_ind, gt_ind = shape_match.nonzero(as_tuple=True)

            # Offsets
            feat_size = gt_xy.new_tensor([[feat_w, feat_h]])
            strides = gt_xy.new_tensor([strides])

            xy_grid = gt_xy[gt_ind] / strides  # grid xy
            xy_grid_inv = feat_size - xy_grid  # inverse just used for fast calculation of neighbor cell validity
            if neighbor == 0:
                pred_x, pred_y = xy_grid.long().T
                anchor_ind = (
                                     pred_y * feat_w + pred_x) * num_base_anchors + base_anchor_ind
            else:
                x_left_ok, y_up_ok = ((xy_grid % 1. < 0.5) & (xy_grid > 1.)).T
                x_right_ok, y_down_ok = (
                        (xy_grid_inv % 1. < 0.5) & (xy_grid_inv > 1.)).T
                if neighbor == 1:
                    neighbor_ok = torch.stack((torch.ones_like(x_left_ok),
                                               x_left_ok,
                                               y_up_ok,
                                               x_right_ok,
                                               y_down_ok))
                    if neighbor_ok.numel() > 0:
                        four_direction_distance = torch.cat(
                            (xy_grid, xy_grid_inv), dim=-1) % 1.
                        direction_mask = (
                                four_direction_distance == four_direction_distance.min(
                            dim=-1).values[:, None])
                        neighbor_ok[1:] = neighbor_ok[1:] & direction_mask.T
                elif neighbor == 2:
                    neighbor_ok = torch.stack((torch.ones_like(x_left_ok),
                                               x_left_ok,
                                               y_up_ok,
                                               x_right_ok,
                                               y_down_ok))
                elif neighbor == 3:
                    xy_upleft_ok = x_left_ok & y_up_ok
                    xy_upright_ok = x_right_ok & y_up_ok
                    xy_downright_ok = x_right_ok & y_down_ok
                    xy_downleft_ok = x_left_ok & y_down_ok
                    neighbor_ok = torch.stack((torch.ones_like(x_left_ok),
                                               x_left_ok,
                                               y_up_ok,
                                               x_right_ok,
                                               y_down_ok,
                                               xy_upleft_ok,
                                               xy_upright_ok,
                                               xy_downright_ok,
                                               xy_downleft_ok))
                else:
                    raise NotImplementedError
                num_offset = neighbor_ok.shape[0]
                gt_ind = gt_ind.repeat((num_offset, 1))[neighbor_ok]
                base_anchor_ind = base_anchor_ind.repeat((num_offset, 1))[
                    neighbor_ok]
                xy_grid_all = \
                    (xy_grid[None, :, :] + neighbor_offset[:num_offset, None,
                                           :])[
                        neighbor_ok]
                pred_x, pred_y = xy_grid_all.long().T
                anchor_ind = (
                                     pred_y * feat_w + pred_x) * num_base_anchors + base_anchor_ind

            indices.append((img_id[gt_ind], anchor_ind, gt_ind))

        return indices

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        # keep as Tensor, so that we can covert to ONNX correctly
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack(
            [shift_xx, shift_yy] + [torch.zeros_like(shift_xx) for _ in
                                    range(3)], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 5) to K shifts (K, 1, 5) to get
        # shifted anchors (K, A, 5), reshape to (K*A, 5)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors
