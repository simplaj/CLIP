import os

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from matplotlib import pyplot as plt


class Angle(torch.utils.data.Dataset):
    """Some Information about Angle"""
    def __init__(self, mode, preprocess):
        super(Angle, self).__init__()
        self.preprocess = preprocess
        self.paths = []
        data = np.load('PD_43.npy', allow_pickle=True)
        labels = data[:, 0]
        names = data[:, 1]
        idxs = self.split(labels=labels.tolist(), k=5, fold_idx=0)
        idx = idxs[mode == 'test']
        self.gen_np_name(labels[idx], names[idx])
        self.process()
        # self.gen_filename(labels[idx], names[idx])

    def __getitem__(self, index):
        path = self.paths[index]
        im = Image.open(path_pair)
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
            
    def gen_np_name(self, lables, names):
        self.labels = []
        self.pairs = []
        for i, name in enumerate(names):
            lable = lables[i]
            prefix = 'ill'
            if isinstance(name, str):
                name = int(name[2:])
                prefix = 'hty'
            id_pair = [(x, y) for x in (0, 2, 4) for y in (1, 3, 5)]
            if os.path.exists(f'angle_data/{prefix}/{name}_0_fore_left.npy'):
                path_pairs1 = [(
                    f'angle_data/{prefix}/{name}_{x}_upper_left.npy',
                    f'angle_data/{prefix}/{name}_{x}_fore_left.npy',
                    f'angle_data/{prefix}/{name}_{y}_upper_right.npy',
                    f'angle_data/{prefix}/{name}_{y}_fore_right.npy',
                    ) for x, y in id_pair]
            else:
                path_pairs1 = [(
                    f'angle_data/{prefix}/{name}_{x}_upper_right.npy',
                    f'angle_data/{prefix}/{name}_{x}_fore_right.npy',
                    f'angle_data/{prefix}/{name}_{y}_upper_left.npy',
                    f'angle_data/{prefix}/{name}_{y}_fore_left.npy',
                    ) for x, y in id_pair]
            self.pairs.extend(path_pairs1)
            self.labels.extend([lable for _ in path_pairs1])
    
    def concat_img(self, path_pair):
        im0 = Image.open(path_pair[0])
        im1 = Image.open(path_pair[1])
        im0_np = np.asarray(im0)
        im1_np = np.asarray(im1)
        im_np = np.concatenate([im0, im1], axis=1)
        im = Image.fromarray(im_np)
        return im
    
    def draw_img(self, path_pair):
        file_name_0 = os.path.basename(path_pair[0])
        file_name_2 = os.path.basename(path_pair[2])
        name = file_name_0.split('_')[0]
        x, y = int(file_name_0.split('_')[1]), int(file_name_2.split('_')[1])
        idxs = [2, 0]
        data = []
        if 'left' in path_pair[0]:
            idxs = [0, 2]
        for idx in idxs:
            data.append([
                np.load(path_pair[idx]),
                np.load(path_pair[idx + 1])
                ])
        y_min = [0, 120]
        y_max = [20, 180]
        x_min = 0
        x_max = 10
        color_map = ['b', 'r']
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), facecolor='black')
        
        for ii, param_list in enumerate(data):
            for i in range(len(param_list)):
                param_x = np.arange(param_list[i].shape[0]) / 25
                axs[i].axis('off')  # 关闭坐标轴
                axs[i].set_facecolor('black')
                axs[i].set_ylim(ymin = y_min[i], ymax = y_max[i])
                axs[i].set_xlim(xmin = x_min, xmax = x_max)
                axs[i].plot(param_x, param_list[i], c=color_map[ii])
        save_path = path_pair[0].replace(file_name_0, f'{name}_{x}_{y}.jpg')
        self.paths.append(save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()

    def process(self):
        for path_pair in self.pairs:
            self.draw_img(path_pair)


if __name__ == '__main__':
    angle = Angle(mode='test', preprocess=None)