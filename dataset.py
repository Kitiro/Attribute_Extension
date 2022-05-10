#!/usr/bin/env python
# coding=utf-8
"""
Author: Kitiro
Date: 2020-11-03 19:44:27
LastEditTime: 2020-11-05 11:39:32
LastEditors: Kitiro
Description: 
FilePath: /zzc/exp/Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/dataset.py
"""
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.io as sio
from sklearn import preprocessing as P


def screen_attr(attr_matrix, ratio=0.2, norm=False):
    std = np.std(attr_matrix, axis=0)
    # generated attributes need to be normalized
    threshold = np.sort(std, axis=0)[int(std.shape[0] * ratio)]
    ret = np.squeeze(attr_matrix[:, np.argwhere(std > threshold)])
    if norm:
        ret = P.scale(ret, axis=0)
    return ret


class DataSet(Dataset):
    def __init__(
        self,
        name="AWA2",
        generative_attribute_num=0,
        screen=False,
        screen_ratio=0.2,
        norm=False,
    ):
        self.name = name
        self.data_dir = f"data/{name}_data"
        self.generative_attribute_num = generative_attribute_num
        self.screen = screen
        self.screen_ratio = screen_ratio
        if name == "CUB123":
            mat = sio.loadmat(os.path.join(self.data_dir, "att_splits.mat"))
            attribute1 = mat["att"]
            
            mat = sio.loadmat(os.path.join(self.data_dir, "res101.mat"))
            
            self.train_feature = mat["train_feature"]
            self.train_label = mat["train_label"].squeeze() 
            self.test_feature_unseen = mat["test_feature_unseen"]
            self.test_label_unseen = mat["test_label_unseen"].squeeze().astype(int)

            self.test_feature_seen = mat["test_feature_seen"]
            self.test_label_seen = mat["test_label_seen"].squeeze().astype(int)

            if norm:
                attribute1 = P.scale(attribute1, axis=0)
                # self.train_feature = P.scale(self.train_feature, axis=1)
                # self.test_feature_unseen = P.scale(self.test_feature_unseen, axis=1)
                # self.test_feature_seen = P.scale(self.test_feature_seen, axis=1)
        else:
            mat_visual = sio.loadmat(os.path.join(self.data_dir, "res101.mat"))
            features, labels = (
                mat_visual["features"].T,
                mat_visual["labels"].astype(int).squeeze() - 1,
            )

            mat_semantic = sio.loadmat(os.path.join(self.data_dir, "att_splits.mat"))

            attribute1 = mat_semantic["att"].T 
            if norm:
                attribute1 = mat_semantic["att"].T
                attribute1 = P.scale(attribute1, axis=0)
                features = P.scale(features, axis=1)

            trainval_loc = mat_semantic["trainval_loc"].squeeze() - 1
            test_seen_loc = mat_semantic["test_seen_loc"].squeeze() - 1
            test_unseen_loc = mat_semantic["test_unseen_loc"].squeeze() - 1

            self.train_feature = features[trainval_loc]  # feature
            self.train_label = labels[trainval_loc].astype(int)  # 23527 training samples。

            # self.train_label_unique = np.unique(self.train_label)

            self.test_feature_unseen = features[test_unseen_loc]  # 7913 测试集中的未见类
            self.test_label_unseen = labels[test_unseen_loc].astype(int)

            self.test_feature_seen = features[test_seen_loc]  # 5882  测试集中的已见类
            self.test_label_seen = labels[test_seen_loc].astype(int)

        self.attr1_dim = (attribute1.shape[0], attribute1.shape[1])

        if self.generative_attribute_num != 0:
            attribute2 = np.load(
                f"generate_attributes/generated_attributes_glove/class_attribute_map_{self.name}.npy"
            )
            attribute2 = attribute2[:, :generative_attribute_num]
            self.attr2_dim = (attribute2.shape[0], attribute2.shape[1])
            if screen:
                attribute1 = screen_attr(attribute1, self.screen_ratio, norm=False)
                attribute2 = screen_attr(attribute2, self.screen_ratio, norm=True)
            attribute = np.hstack((attribute1, attribute2))  # concat generated attributes horizontally
        else:
            # only adpot original attribute
            attribute = attribute1
        self.attribute = attribute
        # self.attr_dim = attribute.shape[0]
        self.attr_dim = (attribute.shape[0], attribute.shape[1])
        self.train_att = attribute[self.train_label]  # 23527*85
        self.test_id_unseen = np.unique(self.test_label_unseen)
        self.test_att_map_unseen = attribute[self.test_id_unseen]

    # for training
    def __getitem__(self, index):

        visual_feature = self.train_feature[index]
        semantic_feature = self.train_att[index]
        label = self.train_label[index]

        return visual_feature, semantic_feature, label

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        info = "-{} dataset: \n\t-Training samples: {}\n\t-Test samples: {}\n\t-Visual Dim: {}\n\t-Attribute Dim: {}x{}".format(
            self.name,
            self.train_feature.shape[0],
            self.test_feature_seen.shape[0] + self.test_feature_unseen.shape[0],
            self.train_feature.shape[1],
            self.attr1_dim[0],
            self.attr1_dim[1],
        )
        if self.generative_attribute_num != 0:
            info += "\n\t-Total Attribute Dim: {}x{}".format(
                self.attr_dim[0], self.attr_dim[1]
            )
        return info


if __name__ == "__main__":
    a = DataSet(
        name="AWA2", generative_attribute_num=100, screen=False, screen_ratio=0.2
    )
    print(a)
