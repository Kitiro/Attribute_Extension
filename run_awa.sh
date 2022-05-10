#!/bin/bash
###
 # @Author: Kitiro
 # @Date: 2020-11-05 23:50:21
 # @LastEditTime: 2021-06-16 23:01:29
 # @LastEditors: Kitiro
 # @Description: 
 # @FilePath: /Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/run_awa.sh
### 

attr_num='0 50 100 150 200 300'
beta='0 0.0001 0.001'

for a in $attr_num
do
    for c in $beta
    do
            python main.py --dataset AWA --attr_num $a --c_w $c --gpu 1 --batch_size 128 --epochs 300 &
    done
    wait
done




