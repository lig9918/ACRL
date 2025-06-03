CUDA_VISIBLE_DEVICES=0 \
python3 test_sysu.py \
-b 64 -a agw -d  sysu_all \
--iters 200 \
--eps 0.6 \
--num-instances 16 \
--logs-dir "/data-sysu/lg10/cvpr23_upload/origin"
