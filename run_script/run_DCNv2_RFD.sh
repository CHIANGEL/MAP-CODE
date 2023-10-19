export CUDA_VISIBLE_DEVICES=0
dataset_name=avazu

python -u code/run.py --pretrain=True \
    --output_dir=outputs/${dataset_name}/RFD/pretrain \
    --dataset_name=${dataset_name} \
    --data_dir=data/${dataset_name} \
    --num_train_epochs=3 \
    --per_gpu_train_batch_size=4096 \
    --per_gpu_eval_batch_size=4096 \
    --learning_rate=1e-3 \
    --lr_sched=cosine \
    --weight_decay=5e-2 \
    \
    --pt_type=RFD \
    --sampling_method=randint \
    --mask_ratio=0.3 \
    --RFD_replace=Unigram \
    \
    --model_name=DCNv2 \
    --embed_size=16 \
    --hidden_size=1000 \
    --num_hidden_layers=3 \
    --num_cross_layers=3 \
    --hidden_dropout_rate=0.0