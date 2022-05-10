

###
 # @Author: Kitiro
 # @Date: 2020-11-05 23:54:24
 # @LastEditTime: 2021-06-21 00:56:45
 # @LastEditors: Kitiro
 # @Description: 
 # @FilePath: /Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/run_flo.sh
### 

attr_num='0 50 100 150 200 300'
alpha='0.0'
beta='0 0.1 0.01 0.001 0.0001'
# for a in $attr_num
# do
#     for j in $alpha
#     do
#         for c in $beta
#         do
#             python main.py --dataset AWA2 --attr_num $a --c_w $c --gpu 1 --batch_size 128  &
#             python main.py --dataset AWA2 --attr_num $a --c_w $c --gpu 1 --batch_size 128  &
#             python main.py --dataset AWA2 --attr_num $a --c_w $c --gpu 1 --batch_size 128  &
#         done
#         wait
#     done
# done


for c in $beta
do
        python main.py --dataset FLO --attr_num 0 --c_w $c --gpu 1 --batch_size 128  
        #python main.py --dataset FLO --attr_num 0 --c_w $c --gpu 1 --batch_size 128  &
done




# seen=0.6976, unseen=0.5481, h=0.6139  Params:0 0.001 6710





