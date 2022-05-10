

###
 # @Author: Kitiro
 # @Date: 2020-11-05 23:54:24
 # @LastEditTime: 2022-01-14 21:10:52
 # @LastEditors: Kitiro
 # @Description: 
 # @FilePath: /Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/run_awa2.sh
### 

attr_num='0 50 100 150 200 300'
alpha='0.0'
beta='0'

for a in $attr_num
do
    for c in $beta
    do
            python main.py --dataset AWA2 --attr_num $a --c_w $c --gpu 1 --batch_size 512  &
            python main.py --dataset AWA2 --attr_num $a --c_w $c --gpu 1 --batch_size 512  &
    done
    wait
done


# seen=0.6976, unseen=0.5481, h=0.6139  Params:0 0.001 6710





