import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image


class Angle(torch.utils.data.Dataset):
    """Some Information about Angle"""
    def __init__(self, mode, preprocess):
        super(Angle, self).__init__()
        self.preprocess = preprocess
        data = np.load('PD_43.npy', allow_pickle=True)
        labels = data[:, 0]
        names = data[:, 1]
        idxs = self.split(labels=labels.tolist(), k=5, fold_idx=0)
        idx = idxs[mode == 'test']
        self.gen_filename(labels[idx], names[idx])

    def __getitem__(self, index):
        path_pair = self.paths[index]
        im = self.concat_img(path_pair)
        return self.preprocess(im), self.labels[index]

    def __len__(self):
        return len(self.labels)

    def split(self, labels, k, fold_idx=0):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]
        return train_idx, valid_idx
    
    def gen_filename(self, lables, names):
        self.labels = []
        self.paths = []
        for i, name in enumerate(names):
            lable = lables[i]
            prefix = 'ill'
            if isinstance(name, str):
                name = int(name[2:])
                prefix = 'health'
            id_pair = [(x, y) for x in (0, 2, 4) for y in (1, 3, 5)]
            id_pair += [(y, x) for x, y in id_pair]
            path_pairs1 = [(f'../pd_img/{prefix}/{name}_{x}_angle.jpg',
                            f'../pd_img/{prefix}/{name}_{y}_angle.jpg')\
                            for x, y in id_pair]
            self.paths.extend(path_pairs1)
            self.labels.extend([lable for _ in path_pairs1])
    
    def concat_img(self, path_pair):
        im0 = Image.open(path_pair[0])
        im1 = Image.open(path_pair[1])
        im0_np = np.asarray(im0)
        im1_np = np.asarray(im1)
        im_np = np.concatenate([im0, im1], axis=1)
        im = Image.fromarray(im_np)
        return im
    
    
if __name__ == '__main__':
    angle = Angle(mode='train')