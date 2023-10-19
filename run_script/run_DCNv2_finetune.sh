export CUDA_VISIBLE_DEVICES=0
dataset_name=avazu
pt_type=RFD

python -u code/run.py --finetune \
    --pretrained_model_path=outputs/avazu/${pt_type}/pretrain/9.model \
    \
    --output_dir=outputs/${dataset_name}/${pt_type}/finetune \
    --dataset_name=${dataset_name} \
    --data_dir=data/${dataset_name} \
    --num_train_epochs=1 \
    --per_gpu_train_batch_size=4096 \
    --per_gpu_eval_batch_size=4096 \
    --learning_rate=1e-3 \
    --lr_sched=const \
    --weight_decay=1e-1 \
    \
    --model_name=DCNv2 \
    --embed_size=16 \
    --hidden_size=1000 \
    --num_hidden_layers=3 \
    --num_cross_layers=3 \
    --hidden_dropout_rate=0.0