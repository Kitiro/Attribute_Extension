

###
 # @Author: Kitiro
 # @Date: 2020-11-05 23:54:18
 # @LastEditTime: 2021-06-18 00:13:49
 # @LastEditors: Kitiro
 # @Description: 
 # @FilePath: /Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/run_cub.sh
### 


attr_num='0 100 200 300'
beta='0 0.0001 0.001'

for a in $attr_num
do
    for c in $beta
    do
            python main.py --dataset CUB --attr_num $a --c_w $c --gpu 1 --batch_size 256 --epochs 500 &
    done
    wait
done