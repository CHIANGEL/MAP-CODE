import subprocess
import sys

# Training args
wandb_name = sys.argv[1]
if wandb_name == 'avazu_x4_core2_emb32':
    dataset_name = 'avazu_x4'
    n_core = '2'
    embed_size = 32
elif wandb_name == 'avazu_x4_core2_emb64':
    dataset_name = 'avazu_x4'
    n_core = '2'
    embed_size = 64
elif wandb_name == 'criteo_x4_core10_emb64':
    dataset_name = 'criteo_x4'
    n_core = '10'
    embed_size = 64
elif wandb_name == 'criteo_x4_core10_emb32':
    dataset_name = 'criteo_x4'
    n_core = '10'
    embed_size = 32
elif wandb_name == 'criteo_x4_core10_emb16':
    dataset_name = 'criteo_x4'
    n_core = '10'
    embed_size = 16
elif wandb_name == 'criteo_x4_core10_emb8':
    dataset_name = 'criteo_x4'
    n_core = '10'
    embed_size = 8
else:
    assert 0
if dataset_name == 'avazu_x4':
    data_dir = '/home/chiangel/data/avazu/avazu_x4'
elif dataset_name == 'criteo_x4':
    data_dir = '/home/chiangel/data/criteo/criteo_x4'
    # data_dir = '../data/criteo/criteo_x4'
else:
    assert 0
seed = 42
split_name = 'x4-full'
epoch = 20
batch_size = 4096
lr = '1e-3'
weight_decay = '1e-1'
lr_sched = 'const'
adam_epsilon, adam_betas = ('1e-8', '0.9,0.999')
warmup = '0.0'

# Model config
model = 'FiBiNet'
hidden_size = 2000
num_hidden_layers = 0
dropout = '0.0'
reduction_ratio = 3
bilinear_type = 'field_interaction'
use_lr = '_NoLr'

# Run the train process
seed = sys.argv[2]
weight_decay = sys.argv[3]
reduction_ratio = sys.argv[4]
for weight_decay in ['1e-1', '5e-2']:
    for seed in [42, 43, 44]:
        subprocess.run(['python', 'train.py',
                        '--use_lr' if use_lr == 'Lr' else '', 
                        f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                        f'hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}_reduce{reduction_ratio}_{bilinear_type}{use_lr}',

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
                        f'--reduction_ratio={reduction_ratio}',
                        f'--bilinear_type={bilinear_type}',
                        ])

# √ criteo_x4_core10_emb16 | wandb seed weight_decay reduction_ratio
    # for weight_decay in ['1e-1', '5e-2']:
    # for seed in [42, 43, 44]:
    # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py criteo_x4_core10_emb16 seed wd 2 & √
    # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py criteo_x4_core10_emb16 seed wd 3 & √
    # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py criteo_x4_core10_emb16 seed wd 4 & √

# √ criteo_x4_core10_emb32 | seed reduction_ratio epoch lr_sched weight_decay
    # for seed in [42, 43, 44]:
    # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py seed 2 20 const 1e-1 & √
    # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py seed 2 5 cosine 1e-1 & √
    # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py seed 3 20 const 1e-1 & √
    # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py seed 3 5 cosine 1e-1 & √
    # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py seed 4 20 const 1e-1 & √
    # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py seed 4 5 cosine 1e-1 & √
    # gpu10: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py seed 2 20 const 5e-2 & √
    # gpu10: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py seed 2 5 cosine 5e-2 & √
    # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py seed 3 20 const 5e-2 & √
    # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py seed 3 5 cosine 5e-2 & √
    # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_fibinet.py seed 4 20 const 5e-2 & √
    # gpu06: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py seed 4 5 cosine 5e-2 & √

# avazu_x4_core2_emb64
    # √ 基于最优模型，重复实验取平均，field_interaction，wd=1e-1 | seed reduction_ratio epoch lr_sched
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py 42 2 20 const & √
        # gpu26: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fibinet.py 42 2 5 cosine & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py 42 3 20 const & √
        # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py 42 3 5 cosine & √
        # gpu26: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py 43 2 20 const & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py 43 2 5 cosine & √
        # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fibinet.py 43 3 20 const & √
        # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py 43 3 5 cosine & √
        # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py 44 2 20 const & √
        # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py 44 2 5 cosine & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py 44 3 20 const & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fibinet.py 44 3 5 cosine & √
        # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fibinet.py 45 2 20 const & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py 45 2 5 cosine & √
        # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_fibinet.py 45 3 20 const & √
        # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_fibinet.py 45 3 5 cosine & √
        # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py 46 2 20 const & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fibinet.py 46 2 5 cosine & √
        # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_fibinet.py 46 3 20 const & √
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fibinet.py 46 3 5 cosine & √

    # √ seed=42，调最优模型 | wd reduction_ratio bilinear_type epoch lr_sched
        # gpu17: CUDA_VISIBLE_DEVICES=0 python -u run_fibinet.py 1e-1 2 field_interaction 20 const & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 python -u run_fibinet.py 5e-2 2 field_interaction 20 const & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 python -u run_fibinet.py 1e-1 3 field_interaction 20 const & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 python -u run_fibinet.py 5e-2 3 field_interaction 20 const & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 python -u run_fibinet.py 1e-1 4 field_interaction 20 const & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 python -u run_fibinet.py 5e-2 4 field_interaction 20 const & √

        # gpu21: CUDA_VISIBLE_DEVICES=0 python -u run_fibinet.py 1e-1 2 field_interaction 5 cosine & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 python -u run_fibinet.py 5e-2 2 field_interaction 5 cosine & √
        # gpu22: CUDA_VISIBLE_DEVICES=0 python -u run_fibinet.py 1e-1 3 field_interaction 5 cosine & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 python -u run_fibinet.py 5e-2 3 field_interaction 5 cosine & √
        # gpu28: CUDA_VISIBLE_DEVICES=0 python -u run_fibinet.py 1e-1 4 field_interaction 5 cosine & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 python -u run_fibinet.py 5e-2 4 field_interaction 5 cosine & √
        