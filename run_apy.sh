

###
 # @Author: Kitiro
 # @Date: 2020-11-05 23:54:10
 # @LastEditTime: 2021-06-24 17:32:17
 # @LastEditors: Kitiro
 # @Description: 
 # @FilePath: /Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/run_apy.sh
### 

attr_num='0 100 200 300'
beta='0 0.0001'
for a in $attr_num
do
    for c in $beta
    do
            python main.py --dataset APY --attr_num $a --c_w $c --gpu 0 --batch_size 128 --epochs 300 &
    done
    wait
done

