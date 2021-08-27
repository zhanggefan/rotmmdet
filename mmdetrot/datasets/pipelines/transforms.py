from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import (Resize, RandomFlip,
                                                 RandomCrop)
from mmdet.datasets.pipelines.compose import Compose as PipelineCompose
from ...ops.geo.geo_utils import rboxes_truncate
import random
import cv2
import numpy as np

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module(force=True)
class MosaicPipeline(object):

    def __init__(self, individual_pipeline, pad_val=0):
        self.individual_pipeline = PipelineCompose(individual_pipeline)
        self.pad_val = pad_val

    def __call__(self, results):
        input_results = results.copy()
        mosaic_results = [results]
        dataset = results['dataset']
        # load another 3 images
        indices = dataset.batch_rand_others(results['_idx'], 3)
        for idx in indices:
            img_info = dataset.data_infos[idx]
            ann_info = dataset.get_ann_info(idx)
            _results = dict(img_info=img_info, ann_info=ann_info, _idx=idx)
            if dataset.proposals is not None:
                _results['proposals'] = dataset.proposals[idx]
            dataset.pre_pipeline(_results)
            mosaic_results.append(_results)

        for idx in range(4):
            mosaic_results[idx] = self.individual_pipeline(mosaic_results[idx])

        shapes = [results['pad_shape'] for results in mosaic_results]
        cxy = max(shapes[0][0], shapes[1][0], shapes[0][1], shapes[2][1])
        canvas_shape = (cxy * 2, cxy * 2, shapes[0][2])

        # base image with 4 tiles
        canvas = dict()
        for key in mosaic_results[0].get('img_fields', []):
            canvas[key] = np.full(canvas_shape, self.pad_val, dtype=np.uint8)
        for i, results in enumerate(mosaic_results):
            h, w = results['pad_shape'][:2]
            # place img in img4
            if i == 0:  # top left
                x1, y1, x2, y2 = cxy - w, cxy - h, cxy, cxy
            elif i == 1:  # top right
                x1, y1, x2, y2 = cxy, cxy - h, cxy + w, cxy
            elif i == 2:  # bottom left
                x1, y1, x2, y2 = cxy - w, cxy, cxy, cxy + h
            elif i == 3:  # bottom right
                x1, y1, x2, y2 = cxy, cxy, cxy + w, cxy + h

            for key in mosaic_results[0].get('img_fields', []):
                canvas[key][y1:y2, x1:x2] = results[key]

            for key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, 0::2] = bboxes[:, 0::2] + x1
                bboxes[:, 1::2] = bboxes[:, 1::2] + y1
                results[key] = bboxes

        output_results = input_results
        output_results['filename'] = None
        output_results['ori_filename'] = None
        output_results['img_fields'] = mosaic_results[0].get('img_fields', [])
        output_results['bbox_fields'] = mosaic_results[0].get(
            'bbox_fields', [])
        for key in output_results['img_fields']:
            output_results[key] = canvas[key]

        for key in output_results['bbox_fields']:
            output_results[key] = np.concatenate(
                [r[key] for r in mosaic_results], axis=0)

        output_results['gt_labels'] = np.concatenate(
            [r['gt_labels'] for r in mosaic_results], axis=0)

        output_results['img_shape'] = canvas_shape
        output_results['ori_shape'] = canvas_shape
        output_results['flip'] = False
        output_results['flip_direction'] = None

        return output_results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'individual_pipeline={self.individual_pipeline}, '
                    f'pad_val={self.pad_val})')
        return repr_str


@PIPELINES.register_module()
class HueSaturationValueJitter(object):

    def __init__(self, hue_ratio=0.5, saturation_ratio=0.5, value_ratio=0.5):
        self.h_ratio = hue_ratio
        self.s_ratio = saturation_ratio
        self.v_ratio = value_ratio

    def __call__(self, results):
        for key in results.get('img_fields', []):
            results[key] = np.ascontiguousarray(results[key])
            img = results[key]
            # random gains
            r = np.array([random.uniform(-1., 1.) for _ in range(3)]) * \
                [self.h_ratio, self.s_ratio, self.v_ratio] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                                 cv2.LUT(val, lut_val))).astype(dtype)
            cv2.cvtColor(
                img_hsv, cv2.COLOR_HSV2BGR,
                dst=results[key])  # no return needed
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hue_ratio={self.h_ratio}, '
                    f'saturation_ratio={self.s_ratio}, '
                    f'value_ratio={self.v_ratio})')
        return repr_str


@PIPELINES.register_module()
class MosaicPipelineR(MosaicPipeline):
    def __call__(self, results):
        input_results = results.copy()
        mosaic_results = [results]
        dataset = results['dataset']
        # load another 3 images
        indices = dataset.batch_rand_others(results['_idx'], 3)
        for idx in indices:
            img_info = dataset.data_infos[idx]
            ann_info = dataset.get_ann_info(idx)
            _results = dict(img_info=img_info, ann_info=ann_info, _idx=idx)
            if dataset.proposals is not None:
                _results['proposals'] = dataset.proposals[idx]
            dataset.pre_pipeline(_results)
            mosaic_results.append(_results)

        for idx in range(4):
            mosaic_results[idx] = self.individual_pipeline(mosaic_results[idx])

        shapes = [results['pad_shape'] for results in mosaic_results]
        cxy = max(shapes[0][0], shapes[1][0], shapes[0][1], shapes[2][1])
        canvas_shape = (cxy * 2, cxy * 2, shapes[0][2])

        # base image with 4 tiles
        canvas = dict()
        for key in mosaic_results[0].get('img_fields', []):
            canvas[key] = np.full(canvas_shape, self.pad_val, dtype=np.uint8)
        for i, results in enumerate(mosaic_results):
            h, w = results['pad_shape'][:2]
            # place img in img4
            if i == 0:  # top left
                x1, y1, x2, y2 = cxy - w, cxy - h, cxy, cxy
            elif i == 1:  # top right
                x1, y1, x2, y2 = cxy, cxy - h, cxy + w, cxy
            elif i == 2:  # bottom left
                x1, y1, x2, y2 = cxy - w, cxy, cxy, cxy + h
            elif i == 3:  # bottom right
                x1, y1, x2, y2 = cxy, cxy, cxy + w, cxy + h

            for key in mosaic_results[0].get('img_fields', []):
                canvas[key][y1:y2, x1:x2] = results[key]

            for key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, 0] = bboxes[:, 0] + x1
                bboxes[:, 1] = bboxes[:, 1] + y1
                results[key] = bboxes

        output_results = input_results
        output_results['filename'] = None
        output_results['ori_filename'] = None
        output_results['img_fields'] = mosaic_results[0].get('img_fields', [])
        output_results['bbox_fields'] = mosaic_results[0].get(
            'bbox_fields', [])
        for key in output_results['img_fields']:
            output_results[key] = canvas[key]

        for key in output_results['bbox_fields']:
            output_results[key] = np.concatenate(
                [r[key] for r in mosaic_results], axis=0)

        output_results['gt_labels'] = np.concatenate(
            [r['gt_labels'] for r in mosaic_results], axis=0)

        output_results['img_shape'] = canvas_shape
        output_results['ori_shape'] = canvas_shape
        output_results['flip'] = False
        output_results['flip_direction'] = None

        return output_results


@PIPELINES.register_module()
class ResizeR(Resize):
    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            s = results['scale_factor']
            assert (s.max() - s.min()) / s.mean() < 0.05, s
            rbboxes = results[key].copy()
            rbboxes[..., :4] *= np.array([s[0], s[1], s.mean(), s.mean()])
            results[key] = rbboxes


@PIPELINES.register_module()
class RandomFlipR(RandomFlip):
    def bbox_flip(self, rbboxes, img_shape, direction):
        assert rbboxes.shape[-1] % 5 == 0
        flipped = rbboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0] = w - rbboxes[..., 0]
            flipped[..., 4] = - rbboxes[..., 4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1] = h - rbboxes[..., 1]
            flipped[..., 4] = - rbboxes[..., 4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0] = w - rbboxes[..., 0]
            flipped[..., 1] = h - rbboxes[..., 1]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped


@PIPELINES.register_module()
class RandomCropR(RandomCrop):
    def __init__(self, **kwargs):
        self.min_rbbox_ratio = kwargs.pop('min_rbbox_ratio', 0.2)
        self.min_rbbox_area = kwargs.pop('min_box_area', 64)
        self.min_box_expand = kwargs.pop('min_box_expand', 20)
        super(RandomCropR, self).__init__(**kwargs)
        self.rbbox2label = {
            'gt_rbboxes': 'gt_labels',
            'gt_rbboxes_ignore': 'gt_labels_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            rbbox_offset = np.array([offset_w, offset_h, 0, 0, 0],
                                    dtype=np.float32)
            rbboxes = results[key] - rbbox_offset

            h, w, _ = img_shape
            limit = [w / 2, h / 2, w, h, 0]
            rbboxes_trunc = rboxes_truncate(rbboxes, limit)

            valid_inds = ((rbboxes_trunc[:, 2:4].prod(axis=-1) / (
                    rbboxes[:, 2:4].prod(
                        axis=-1) + 1e-7)) > self.min_rbbox_ratio)
            valid_inds = (valid_inds & (rbboxes_trunc[:, 2:4].prod(
                axis=-1) >= self.min_rbbox_area))
            valid_inds = (valid_inds & (rbboxes_trunc[:, 2:4].max(
                axis=-1) >= self.min_box_expand))
            
            rbboxes = rbboxes_trunc.astype(np.float32)

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = rbboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results
