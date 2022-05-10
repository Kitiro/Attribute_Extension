#!/usr/bin/env python
# coding=utf-8
"""
Author: Kitiro
Date: 2020-11-03 17:32:57
LastEditTime: 2020-11-06 00:04:17
LastEditors: Kitiro
Description: 
FilePath: /zzc/exp/Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/main.py
"""

import numpy as np
import torch
import os
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
from torchsummary import summary
import json
import random

# Author Defined
from model import MyModel, APYModel, CUBModel
from utils import (
    compute_accuracy,
    CenterLoss,
    set_seed,
    cal_mean_feature,
    weights_init,
    export_log,
    plot_img,
    EarlyStopping,
)
from dataset import DataSet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="AWA2", help="dataset for experiments"
)
parser.add_argument(
    "--attr_num", type=int, default=0, help="num of generative attribute for training"
)
parser.add_argument("--output", type=str, default="./output/", help="Output directory")

parser.add_argument("--alpha", type=float, default="0.0", help="Cluster loss weight")
parser.add_argument("--beta", type=float, default="0.0001", help="Center loss weight")

parser.add_argument("--seed", type=int, default=5)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument(
    "--screen", action="store_true", default=False, help="Whether to screen attributes"
)
parser.add_argument("--gpu", default="0", help="GPU")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate")
parser.add_argument("--batch_size", type=int, default=50, help="Batch Size")

args = parser.parse_args()


def eval(model, loader):

    mse_loss = nn.MSELoss()
    with torch.set_grad_enabled(False):

        total_num = 0
        running_loss = 0.0

        # Iterate over data.
        for visual_batch, attr_batch, label_batch in loader:
            visual_batch = visual_batch.float().cuda()

            attr_batch = (
                attr_batch.float()
                .reshape(visual_batch.shape[0], 1, 1, attr_batch.shape[1])
                .cuda()
            )

            out_visual = model(attr_batch)  # semantic -> visual space

            loss = mse_loss(out_visual, visual_batch)

            # statistics loss and acc every epoch
            running_loss += loss.item() * visual_batch.shape[0]

            total_num += visual_batch.shape[0]

    return running_loss / total_num

def train_and_test(loader, dataset, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    visual_feature_dim = dataset.train_feature.shape[1]
    semantic_feature_dim = dataset.attr_dim[1]
    class_num = dataset.attr_dim[0]
    if args.dataset == 'APY':
        model = APYModel(semantic_feature_dim, visual_feature_dim)
    elif args.dataset == 'CUB':
        model = CUBModel(semantic_feature_dim, visual_feature_dim)
    else:
        model = MyModel(semantic_feature_dim, visual_feature_dim)
    model.cuda()
    model.apply(weights_init)
    summary(model, input_size=(1, 1, semantic_feature_dim))

    mse_loss = nn.MSELoss()
    center_loss = CenterLoss(
        num_classes=class_num, feat_dim=visual_feature_dim, use_gpu=True
    )

    best_zsl = 0.0
    best_h = 0.0
    h_line = ""
    patience = 7 
    delta = 0.1
    
    all_attr = (
        torch.Tensor(dataset.attribute)
        .reshape(class_num, 1, 1, semantic_feature_dim)
        .float()
        .cuda()
    )
    cluster = np.load(f"data/cluster/{args.dataset}_cluster.npy")
    # hyperparameters
    alpha = args.alpha  # 0.2-64.4; 0.5 - 64.9 ; 0.7-70.3;0.8-72.2; 1-68.29.
    beta = args.beta  # for center_loss. 0.003 in paper
    lr = args.lr
    lr_cl = 0.5  # 0.5 in paper

    postfix = "Alpha-{}_Beta-{}_GenAttr-{}".format(args.alpha, args.beta, args.attr_num)
    if args.screen:  postfix += '_screen'
    params = list(model.parameters()) + list(
        center_loss.parameters()
    )  # joint learning with center loss
    opt = torch.optim.Adam(
        params, lr=lr, weight_decay=1e-2
    )  # here lr is the overall learning rate
    scheduler = lr_scheduler.MultiStepLR(
        opt, milestones=list(range(20, 300, 20)), gamma=0.9
    )

    history = {
        "Acc_ZSL": [],
        "Loss_attr": [],
        "Loss_cluster": [],
        "Loss_center": [],
        "Loss_Total": [],
    }
    log_head = "Original_attr: {}\tGenerated_attr: {}\tIs_screen: {}\tScreen_ratio: {}\tTotal_attr: {}\n".format(
        dataset.attr_dim[0],
        dataset.generative_attribute_num,
        dataset.screen,
        dataset.screen_ratio,
        dataset.attr_dim[0],
    )
    log_head += "Epoch\tAcc_ZSL\tBest_Acc\tLoss_attr\tLoss_cluster(alpha)\tLoss_center(beta)\tTotal_Loss\n"
    log_path = os.path.join(args.output, args.dataset, "LOG", f"Log_{postfix}.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    export_log(log_head, log_path)

    with torch.set_grad_enabled(True):
        for epoch in range(args.epochs):
            model.train()
            for visual_batch, attr_batch, label_batch in loader:
                visual_batch = visual_batch.float().cuda()

                attr_batch = (
                    attr_batch.float()
                    .reshape(visual_batch.shape[0], 1, 1, semantic_feature_dim)
                    .cuda()
                )
                opt.zero_grad()

                out_visual = model(attr_batch)  # semantic -> visual space

                loss_attr = mse_loss(out_visual, visual_batch)
                if epoch >= 10:
                    which_cluster = cluster[label_batch]
                    cluster_feature_batch = cluster_features[which_cluster]
                    loss_cluster = alpha * mse_loss(
                        visual_batch, torch.tensor(cluster_feature_batch).float().cuda()
                    )
                    loss_center = beta * center_loss(visual_batch, label_batch.cuda())
                    loss = loss_attr + loss_cluster + loss_center
                    opt.zero_grad()
                    loss.backward()

                    # by doing so, weight_cent would not impact on the learning of centers
                    for param in center_loss.parameters():
                        param.grad.data *= lr_cl / (beta)
                    opt.step()
                else:
                    loss_center = beta * center_loss(visual_batch, label_batch.cuda())
                    loss = loss_attr + loss_center

                    opt.zero_grad()
                    loss.backward()
                    for param in center_loss.parameters():
                        param.grad.data *= lr_cl / (beta)
                    opt.step()

            scheduler.step()
            model.eval()

            # early_stopping(eval(model, eval_loader), model)

            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            
            all_feature = model(all_attr)
            cluster_features = cal_mean_feature(
                cluster, all_feature.cpu().detach().numpy()
            )

            acc_zsl = compute_accuracy(
                model,
                dataset.test_att_map_unseen,
                dataset.test_feature_unseen,
                dataset.test_id_unseen,
                dataset.test_label_unseen,
            )
            acc_seen_gzsl = compute_accuracy(
                model,
                dataset.attribute,
                dataset.test_feature_seen,
                np.arange(class_num),
                dataset.test_label_seen,
            )
            acc_unseen_gzsl = compute_accuracy(
                model,
                dataset.attribute,
                dataset.test_feature_unseen,
                np.arange(class_num),
                dataset.test_label_unseen,
            )
            H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)

            if acc_zsl > best_zsl:
                best_zsl = acc_zsl
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.output, "{}/Model_{}.pth".format(args.dataset, postfix)
                    ),
                )
                torch.save(
                    center_loss.state_dict(),
                    os.path.join(
                        args.output,
                        "{}/CenterLoss_{}.pth".format(args.dataset, postfix),
                    ),
                )

            if H > best_h:
                best_h = H
                h_line = "gzsl: seen=%.4f, unseen=%.4f, h=%.4f" % (
                    acc_seen_gzsl,
                    acc_unseen_gzsl,
                    H,
                )

            print("Epoch:", epoch, "--------")
            print("zsl:", acc_zsl)
            print("best_zsl:", best_zsl)

            if epoch < 10:
                print(
                    "Total-loss:{}\nloss_attr:{}, loss_center:{}".format(
                        loss.item(), loss_attr.item(), loss_center.item()
                    )
                )

                history["Loss_cluster"].append(0)
                log_info = "{}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\n".format(
                    epoch,
                    acc_zsl,
                    best_zsl,
                    loss_attr.item(),
                    0,
                    loss_center.item(),
                    loss.item(),
                )
            else:
                print(
                    "Total-loss:{}\nloss_attr:{}, loss_cluster:{}, loss_center:{}".format(
                        loss.item(),
                        loss_attr.item(),
                        loss_cluster.item(),
                        loss_center.item(),
                    )
                )
                history["Loss_cluster"].append(loss_cluster.item())
                log_info = "{}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t\t{:4f}\t\t{:4f}\n".format(
                    epoch,
                    acc_zsl,
                    best_zsl,
                    loss_attr.item(),
                    loss_cluster.item(),
                    loss_center.item(),
                    loss.item(),
                )

            print("lr:", opt.param_groups[0]["lr"])
            print(
                "gzsl: seen=%.4f, unseen=%.4f, h=%.4f"
                % (acc_seen_gzsl, acc_unseen_gzsl, H)
            )
            print(h_line)

            history["Acc_ZSL"].append(acc_zsl)
            history["Loss_attr"].append(loss_attr.item())
            history["Loss_center"].append(loss_center.item())
            history["Loss_Total"].append(loss.item())

            export_log(log_info, log_path)

            plot_img(history, os.path.join(args.output, args.dataset), postfix)
            if acc_zsl < best_zsl-delta:
                patience -= 1
                if patience == 0: 
                    print('Early Stopping')
                    break

def main():
    print("Running parameters:")
    print(json.dumps(vars(args), indent=4, separators=(",", ":")))
    dataset = DataSet(
        name=args.dataset,
        generative_attribute_num=args.attr_num,
        screen=args.screen,
        screen_ratio=0.2,
    )

    # 设置随机数种子
    if args.seed != -1:
        set_seed(args.seed)
    dataset_train = TensorDataset(
        torch.from_numpy(dataset.train_feature),
        torch.from_numpy(dataset.train_att),
        torch.from_numpy(dataset.train_label),
    )
    train_loader = DataLoader(
        dataset=dataset_train, batch_size=100, shuffle=True, num_workers=0
    )
    # data_length = dataset.train_feature.shape[0]
    # eval_ratio = 0.2
    # dataset_eval = TensorDataset(
    #     torch.from_numpy(
    #         dataset.train_feature[
    #             random.sample(range(data_length), int(eval_ratio * data_length))
    #         ]
    #     ),
    #     torch.from_numpy(
    #         dataset.train_att[
    #             random.sample(range(data_length), int(eval_ratio * data_length))
    #         ]
    #     ),
    #     torch.from_numpy(
    #         dataset.train_label[
    #             random.sample(range(data_length), int(eval_ratio * data_length))
    #         ]
    #     ),
    # )
    # eval_loader = DataLoader(
    #     dataset=dataset_eval, batch_size=100, shuffle=False, num_workers=0
    # )
    train_and_test(train_loader, dataset, args)


if __name__ == "__main__":
    main()
