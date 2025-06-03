CUDA_VISIBLE_DEVICES=0 \
python3 test_regdb.py \
-b 64 -a agw -d  regdb_rgb \
--iters 100 \
--eps 0.6 --num-instances 16 \
--logs-dir "D:\data\pythonProject\Regdb-test\regdb_base\origin\regdb"



python test_regdb.py -b 64 -a agw -d  regdb_rgb --iters 100 --eps 0.6 --num-instances 16 --logs-dir "D:\data\pythonProject\Regdb-test\regdb_base\origin\regdb"