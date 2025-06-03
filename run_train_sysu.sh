CUDA_VISIBLE_DEVICES=2,3 \
python train_sysu.py -b 192 -a agw -d  sysu_all \
--num-instances 16 \
--data-dir "/opt/data/private/lg/USL-VI-ReID/SYSU-MM01" \
--logs-dir "/opt/data/private/lg/DMDA/origin" \