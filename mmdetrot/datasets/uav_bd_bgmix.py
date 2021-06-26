from mmdet.datasets import DATASETS, build_dataset
from .uav_bd import UAVBD
import numpy as np


@DATASETS.register_module()
class UAVBD_BGMix(UAVBD):
    def __init__(self, bg_ratio, bg_dataset_conf, **kwargs):
        super(UAVBD_BGMix, self).__init__(**kwargs)
        self.bg_dataset = build_dataset(bg_dataset_conf,
                                        default_args=dict(pipeline=[]))
        bg_len = round(bg_ratio * len(self.data_infos))
        bg_flag = np.zeros(bg_len, dtype=np.uint8)
        self.data_infos += [None] * bg_len
        self.flag = np.concatenate((self.flag, bg_flag))

    def get_ann_info(self, idx):
        if self.data_infos[idx] is None:
            return None
        else:
            return super(UAVBD_BGMix, self).get_ann_info(idx)

    def prepare_test_img(self, idx):
        raise NotImplementedError

    def pre_pipeline(self, results):
        if results['img_info'] is None:
            idx = np.random.randint(len(self.bg_dataset))
            results['img_info'] = self.bg_dataset.data_infos[idx]
            ann_info = self.bg_dataset.get_ann_info(idx)

            ann_info['bboxes'] = np.zeros_like(ann_info['bboxes'],
                                               shape=(0, 5))
            ann_info['labels'] = np.zeros_like(ann_info['labels'], shape=(0,))
            ann_info['bboxes_ignore'] = np.zeros_like(ann_info['bboxes'],
                                                      shape=(0, 5))
            ann_info['labels_ignore'] = np.zeros_like(ann_info['labels'],
                                                      shape=(0,))

            results['ann_info'] = ann_info
            results['img_prefix'] = self.bg_dataset.img_prefix
            results['seg_prefix'] = self.bg_dataset.seg_prefix
            results['proposal_file'] = self.bg_dataset.proposal_file
            results['bbox_fields'] = []
            results['mask_fields'] = []
            results['seg_fields'] = []
            results['dataset'] = self
        else:
            super(UAVBD_BGMix, self).pre_pipeline(results)
