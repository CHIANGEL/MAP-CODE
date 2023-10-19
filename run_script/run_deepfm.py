import sys
import subprocess
import utils

# Training args
wandb_name = sys.argv[1]
dataset_name, n_core, embed_size = utils.parse_wandb_name(wandb_name)
data_dir = utils.get_data_dir(dataset_name)
seed = 42
split_name = 'x4-full'
epoch = 20
batch_size = 8192
lr = '1e-3'
weight_decay = '1e-1'
lr_sched = 'const'
adam_epsilon, adam_betas = ('1e-8', '0.9,0.999')
warmup = '0.0'

# Model config
model = 'DeepFM'
hidden_size = 1000
num_hidden_layers = 3
dropout = '0.0'

# Run the train process
seed = sys.argv[2]
weight_decay = sys.argv[3]
hidden_size = sys.argv[4]
num_hidden_layers = sys.argv[5]
dropout = sys.argv[6]
for batch_size in [4096, 8192]:
    for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42]:
        for seed in [42]:
            subprocess.run(['python', 'train.py',
                            f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + \
                            f'hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}',
                            
                            f'--seed={seed}',
                            f'--n_core={n_core}',
                            f'--wandb_name={wandb_name}',
                            f'--dataset_name={dataset_name}',
                            f'--split_name={split_name}',
                            f'--num_train_epochs={epoch}',
                            f'--per_gpu_train_batch_size={batch_size}',
                            f'--per_gpu_eval_batch_size={batch_size}',
                            f'--learning_rate={lr}',
                            f'--lr_sched={lr_sched}',
                            f'--weight_decay={weight_decay}',
                            f'--adam_epsilon={adam_epsilon}',
                            f'--adam_betas={adam_betas}',
                            f'--warmup_ratio={warmup}',
                            f'--data_dir={data_dir}',

                            f'--model_name={model}',
                            f'--embed_size={embed_size}',
                            f'--hidden_size={hidden_size}',
                            f'--num_hidden_layers={num_hidden_layers}',
                            f'--hidden_dropout_rate={dropout}',
                            ])

# criteo_x4_core10_emb16 | wandb, seed, wd, hidden_size, hidden_layers, dropout
    # gpu03: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 1000 3  0.0 & √
    # gpu03: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 1000 5  0.0 & √
    # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 1000 6  0.0 & √
    # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 1000 9  0.0 & √
    # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 1000 12 0.0 & √
    # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 2000 3  0.0 & √
    # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 2000 6  0.0 & √
    # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 2000 9  0.0 & √
    # gpu33: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb16 seed wd 2000 12 0.0 & √

# criteo_x4_core10_emb64 | wandb, seed, wd, hidden_size, hidden_layers, dropout
    # gpu26: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 1000 3  0.0 & √
    # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 1000 5  0.0 & √
    # gpu: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 1000 6  0.0 &
    # gpu: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 1000 9  0.0 &
    # gpu37: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 1000 12 0.0 & √
    # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 2000 3  0.0 & √
    # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 2000 6  0.0 & √
    # gpu: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 2000 9  0.0 &
    # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_deepfm.py criteo_x4_core10_emb64 seed wd 2000 12 0.0 & √