from mmdet.datasets import CustomDataset
import numpy as np


def pre_pipeline(self, results):
    results['img_prefix'] = self.img_prefix
    results['seg_prefix'] = self.seg_prefix
    results['proposal_file'] = self.proposal_file
    results['bbox_fields'] = []
    results['mask_fields'] = []
    results['seg_fields'] = []
    results['dataset'] = self


def batch_rand_others(self, idx, batch):
    """Get a batch of random index from the same group as the given
    index."""
    mask = (self.flag == self.flag[idx])
    mask[idx] = False
    pool = np.where(mask)[0]
    if len(pool) == 0:
        return np.array([idx] * batch)
    if len(pool) < batch:
        return np.random.choice(pool, size=batch, replace=True)
    return np.random.choice(pool, size=batch, replace=False)


def prepare_train_img(self, idx):
    """Get training data and annotations after pipeline.

    Args:
        idx (int): Index of data.

    Returns:
        dict: Training data and annotation after pipeline with new keys \
            introduced by pipeline.
    """

    img_info = self.data_infos[idx]
    ann_info = self.get_ann_info(idx)
    results = dict(img_info=img_info, ann_info=ann_info, _idx=idx)
    if self.proposals is not None:
        results['proposals'] = self.proposals[idx]
    self.pre_pipeline(results)
    return self.pipeline(results)


CustomDataset.pre_pipeline = pre_pipeline
CustomDataset.batch_rand_others = batch_rand_others
CustomDataset.prepare_train_img = prepare_train_img
