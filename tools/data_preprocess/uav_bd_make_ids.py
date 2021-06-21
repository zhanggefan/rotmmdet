import glob
import os.path as osp


def make_ids(division):
    ids = glob.glob(f'data/uav-bd/{division}/images/*')
    ext = set([*map(lambda x: osp.splitext(x)[-1], ids)])
    ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], ids)]

    assert ext == {'.jpg'}
    ids_str = '\n'.join(ids)
    with open(f'data/uav-bd/{division}.txt', 'w') as f:
        f.write(ids_str)


if __name__ == '__main__':
    make_ids('train')
    make_ids('val')
    make_ids('test')
