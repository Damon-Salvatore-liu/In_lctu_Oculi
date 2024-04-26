# 读取不了数据，暂时先不管

import pickle, os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import sys
sys.path.append('..')
import utils.img_utils.proc_img as ulib
import utils.x_utils as xutil


class SeqDataset(Dataset):

    def __init__(self, anno_path, data_dir, max_seq_len=20):
        self.anno_path = anno_path
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

        # Load annotations
        with open(self.anno_path, 'rb', encoding='MacRoman') as f:
            self.annos = pickle.load(f)


    def _shuffle(self):
        random_idx = np.random.permutation(self.data_num)
        tmp = [self.annos[i] for i in random_idx]
        self.annos = tmp

    def _load_anno(self, anno_path):
        anno_keys = []
        anno_attr = []
        with open(anno_path, 'rb') as f:
            annos = Pickle.load(f)
        for key, val in annos.iteritems():
            anno_keys.append(key)
            anno_attr.append(val)
        return anno_keys, anno_attr

    def get_batch(self, batch_idx, size=None):
        if batch_idx >= self.batch_num:
            raise ValueError("Batch idx must be in range [0, {}].".format(self.batch_num - 1))

        # Get start and end image index ( counting from 0 )
        start_idx = batch_idx * self.batch_size
        idx_range = []
        for i in range(self.batch_size):
            idx_range.append((start_idx + i) % self.data_num)

        print('batch index: {}, counting from 0'.format(batch_idx))

        seq_tensor = []
        len_list = []
        scores_list = []
        label_list = []
        seq_name_list = []
        state_list = []
        for i in idx_range:
            seq, real_len, scores, state, label, seq_name = self.get_one_in_batch(i, size)
            seq_tensor.append(seq)
            len_list.append(real_len)
            scores_list.append(scores)
            label_list.append(label)
            seq_name_list.append(seq_name)
            state_list.append(state)

        seq_tensor = np.array(seq_tensor)
        len_list = np.array(len_list)
        scores_list = np.array(scores_list)
        state_list = np.array(state_list)
        label_list = np.array(label_list)

        if batch_idx == self.batch_num - 1:
            if self.is_shuffle:
                self._shuffle()
        return seq_tensor, len_list, scores_list, state_list, label_list, seq_name_list

    def get_one_in_batch(self, index, size):
        # seq name
        seq_name, anno = self.annos[index]
        seq_dir = Path(self.data_dir) / seq_name
        imgs_list = sorted(list(seq_dir.glob('*.png')))
        seq = []
        for i, im_path in enumerate(imgs_list):
            im = cv2.imread(str(im_path))
            seq.append(im)

        real_len = len(seq)
        scores = anno['scores']
        state = anno['labels']
        label = np.int32(anno['is_blinked'])
        # Check consistency
        assert real_len == len(scores)
        # Is augment?
        if self.is_augment:
            seq = ulib.aug(seq, color_rng=[0.8, 1.2])

        # Padding and resize to out
        seq = ulib.resize(seq, size)

        # Padding same dims matrix to seq to make sure it can be run by batch
        seq = xutil.pad_to_max_len(seq, self.max_seq_len, pad=np.zeros(seq[0].shape, dtype=np.int32))
        scores = xutil.pad_to_max_len(list(scores), self.max_seq_len, pad=0)
        state = xutil.pad_to_max_len(list(state), self.max_seq_len, pad=0)
        return seq, real_len, scores, state, label, seq_name

if __name__ == "__main__":
    with open('../datas/lrcn/train.p', 'r', encoding='ansi') as f:
        data = f.read()
        print(data)

    data = bytes.fromhex(data.decode('utf-8').replace(' ', ''))
    
    # test = SeqDataset(
    #     anno_path='../datas/lrcn/train1.p',
    #     data_dir='../datas/lrcn/',
    # )
    # print(test.annos)
    # print('完成')