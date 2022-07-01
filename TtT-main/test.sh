conda activate python37

gamma=0.5
dname="HybirdSet"  # "SIGHAN15" "HybirdSet", "TtTSet"
dpath="./data/"$dname
cpath="./ckpt/"$dname"_"$gamma"/lyy_test1"
model="epoch_5_dev_f1_0.936"

tpath="./test"
#mkdir -p $cpath
#mkdir -p $tpath

python -u test.py \
    --ckpt_path $cpath/$model \
    --test_data $dpath/test.txt \
    --out_path $tpath/$model.txt \
    --gpu_id 0 \
    --max_len 128 \
