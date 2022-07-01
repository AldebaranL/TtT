conda activate TtT
gamma=0.5
dname="HybirdSet"  # "SIGHAN15" "HybirdSet", "TtTSet"
dpath="./data/"$dname
bpath="./model/bert/"
#cpath="./ckpt/"$dname"_"$gamma"/"
cpath="./ckpt/"$dname"_"$gamma"/lyy_test2"
mkdir -p $cpath
#python -u main.py \
nohup python -u ./main.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab.txt \
    --train_data $dpath/train.txt \
    --dev_data $dpath/dev.txt\
    --test_data $dpath/test.txt\
    --batch_size 50 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 5 \
    --gpu_id 0 \
    --print_every 100 \
    --save_every 2000 \
    --fine_tune \
    --loss_type FC_FT_CRF\
    --gamma $gamma \
    --model_save_path $cpath \
    --prediction_max_len 128 \
    --dev_eval_path $cpath/dev_pred.txt \
    --final_eval_path $cpath/dev_eval.txt \
    --plot_path $cpath/plot \
    --l2_lambda 1e-5 \
    --training_max_len 128\
    >$cpath/log.txt 2>&1 & \