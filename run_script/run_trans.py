import sys
import subprocess

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
model = 'trans'
num_hidden_layers = 2
num_attn_heads = 1
dropout = '0.0'
norm_first = '_postNorm'
intermediate_size = 4 * embed_size
reduction = 'fc'
use_lr = 'NoLr'
dnn_size = 1000
num_dnn_layers = 0

# Run the process
seed = sys.argv[2]
weight_decay = sys.argv[3]
epoch, lr_sched = sys.argv[4], sys.argv[5]
num_hidden_layers = sys.argv[6]
norm_first = sys.argv[7]
for norm_first in ['_postNorm', '_preNorm']:
    for seed in [42, 43, 44]:
        subprocess.run(['python', 'train.py', '--batch_first', 
                        '--norm_first' if norm_first == '_preNorm' else '',
                        '--use_lr' if use_lr == 'Lr' else '', 
                        f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                        f'tl{num_hidden_layers}_th{num_attn_heads}_td{dropout}{norm_first}_FFN{intermediate_size}_{reduction}Reduction_{use_lr}_ds{dnn_size}_dl{num_dnn_layers}',

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
                        f'--hidden_size={embed_size}',
                        f'--num_hidden_layers={num_hidden_layers}',
                        f'--num_attn_heads={num_attn_heads}',
                        f'--intermediate_size={intermediate_size}',
                        f'--output_reduction={reduction}',
                        f'--hidden_dropout_rate={dropout}',
                        f'--dnn_size={dnn_size}',
                        f'--num_dnn_layers={num_dnn_layers}',
                        ])

# √ criteo_x4_core10_emb16 | wandb seed weight_decay epoch lr_sched num_hidden_layers norm_first
    # for norm_first in ['_postNorm', '_preNorm']:
    # for seed in [42, 43, 44]:
    # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 1e-1 20 const 3  norm_first & √
    # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 1e-1 20 const 6  norm_first & √
    # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 1e-1 20 const 9  norm_first & √
    # gpu37: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 1e-1 20 const 12 norm_first & √
    # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 5e-2 20 const 3  norm_first & √
    # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 5e-2 20 const 6  norm_first & √
    # gpu27: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 5e-2 20 const 9  norm_first & √
    # gpu37: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py criteo_x4_core10_emb16 seed 5e-2 20 const 12 norm_first & √

# √ criteo_x4_core10_emb32 | seed weight_decay epoch lr_sched num_hidden_layers norm_first
    # for norm_first in ['_postNorm', '_preNorm']:
    # for seed in [42, 43, 44]:
    # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 20 const 3  norm_first & √
    # gpu41: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 20 const 6  norm_first & √
    # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 20 const 9  norm_first & √
    # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 20 const 12 norm_first & √
    # gpu10: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 5 cosine 3  norm_first & √
    # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 5 cosine 6  norm_first & √
    # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 5 cosine 9  norm_first & √
    # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py seed 1e-1 5 cosine 12 norm_first & √
    # gpu10: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 20 const 3  norm_first & √
    # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 20 const 6  norm_first & √
    # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 20 const 9  norm_first & √
    # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 20 const 12 norm_first & √
    # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 5 cosine 3  norm_first & √
    # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 5 cosine 6  norm_first & √
    # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 5 cosine 9  norm_first & √
    # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py seed 5e-2 5 cosine 12 norm_first & √

# avazu_x4_core2_emb64
    # √ 重复实验取平均，wd=1e-1，2层 | seed norm_first epoch lr_sched reduction
        # √ reduction='fc'
            # gpu03: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 20 const fc & √
            # gpu03: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 5 cosine fc & √
            # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  20 const fc & √
            # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  5 cosine fc & √
            # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 20 const fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 5 cosine fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  20 const fc & √
            # gpu06: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  5 cosine fc & √
            # gpu06: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 20 const fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 5 cosine fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  20 const fc & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  5 cosine fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 20 const fc & √
            # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 5 cosine fc & √
            # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  20 const fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  5 cosine fc & √
            # gpu26: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 20 const fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 5 cosine fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  20 const fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  5 cosine fc & √
        # √ reduction='mean,fc'
            # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 20 const mean,fc & √
            # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 5 cosine mean,fc & √
            # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  20 const mean,fc & √
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  5 cosine mean,fc & √
            # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 20 const mean,fc & √
            # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 5 cosine mean,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  20 const mean,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  5 cosine mean,fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 20 const mean,fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 5 cosine mean,fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  20 const mean,fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  5 cosine mean,fc & √
            # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 20 const mean,fc & √
            # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 5 cosine mean,fc & √
            # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  20 const mean,fc & √
            # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  5 cosine mean,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 20 const mean,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 5 cosine mean,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  20 const mean,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  5 cosine mean,fc & √
        # √ reduction='sum,fc'
            # gpu03: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 20 const sum,fc & √
            # gpu03: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 5 cosine sum,fc & √
            # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  20 const sum,fc & √
            # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  5 cosine sum,fc & √
            # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 20 const sum,fc & √
            # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 5 cosine sum,fc & √
            # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  20 const sum,fc & √
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  5 cosine sum,fc & √
            # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 20 const sum,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 5 cosine sum,fc & √
            # gpu06: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  20 const sum,fc & √
            # gpu06: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  5 cosine sum,fc & √
            # gpu13: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 20 const sum,fc & √
            # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 5 cosine sum,fc & √
            # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  20 const sum,fc & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  5 cosine sum,fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 20 const sum,fc & √
            # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 5 cosine sum,fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  20 const sum,fc & √
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  5 cosine sum,fc & √
        # √ reduction='attn,fc'
            # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 20 const attn,fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 42 _postNorm 5 cosine attn,fc & √
            # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  20 const attn,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 42 _preNorm  5 cosine attn,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 20 const attn,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py 43 _postNorm 5 cosine attn,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  20 const attn,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_trans.py 43 _preNorm  5 cosine attn,fc & √
            # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 20 const attn,fc & √
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 44 _postNorm 5 cosine attn,fc & √
            # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  20 const attn,fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 44 _preNorm  5 cosine attn,fc & √
            # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 20 const attn,fc & √
            # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 45 _postNorm 5 cosine attn,fc & √
            # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  20 const attn,fc & √
            # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 45 _preNorm  5 cosine attn,fc & √
            # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 20 const attn,fc & √
            # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 46 _postNorm 5 cosine attn,fc & √
            # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  20 const attn,fc & √
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 46 _preNorm  5 cosine attn,fc & √

    # √ 调最优参数 | num_hidden_layers norm_first epoch lr_sched weight_decay
        # √ layer=2
            # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 2 _postNorm 20 const 1e-1 & √
            # gpu03: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 2 _postNorm 20 const 5e-2 & √
            # gpu03: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 2 _postNorm 20 const 0.0  & √
            # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 2 _postNorm 5 cosine 1e-1 & √
            # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 2 _postNorm 5 cosine 5e-2 & √
            # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 2 _postNorm 5 cosine 0.0  & √
            # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 2 _preNorm 20 const 1e-1 & √
            # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 2 _preNorm 20 const 5e-2 & √
            # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 2 _preNorm 20 const 0.0  & √
            # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 2 _preNorm 5 cosine 1e-1 & √
            # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 2 _preNorm 5 cosine 5e-2 & √
            # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 2 _preNorm 5 cosine 0.0  & √
        # √ layer=3
            # gpu26: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 3 _postNorm 20 const 1e-1 & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 3 _postNorm 20 const 5e-2 & √
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 3 _postNorm 5 cosine 1e-1 & √
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 3 _postNorm 5 cosine 5e-2 & √
            # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 3 _preNorm  20 const 1e-1 & √
            # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 3 _preNorm  20 const 5e-2 & √
            # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 3 _preNorm  5 cosine 1e-1 & √
            # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 3 _preNorm  5 cosine 5e-2 & √
        # √ layer=5
            # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 5 _postNorm 20 const 1e-1 & √
            # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 5 _postNorm 20 const 5e-2 & √
            # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 5 _postNorm 20 const 0.0  & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 5 _postNorm 5 cosine 1e-1 & √
            # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 5 _postNorm 5 cosine 5e-2 & √
            # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 5 _postNorm 5 cosine 0.0  & √
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 5 _preNorm 20 const 1e-1 & √
            # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 5 _preNorm 20 const 5e-2 & √
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 5 _preNorm 20 const 0.0  & √
            # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 5 _preNorm 5 cosine 1e-1 & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 5 _preNorm 5 cosine 5e-2 & √
            # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 5 _preNorm 5 cosine 0.0  & √
        # √ layer=6
            # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 6 _postNorm 20 const 1e-1 & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py 6 _postNorm 20 const 5e-2 & √
            # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_trans.py 6 _postNorm 5 cosine 1e-1 & √
            # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 6 _postNorm 5 cosine 5e-2 & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py 6 _preNorm  20 const 1e-1 & √
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 6 _preNorm  20 const 5e-2 & √
            # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 6 _preNorm  5 cosine 1e-1 & √
            # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 6 _preNorm  5 cosine 5e-2 & √
        # √ layer=8
            # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 8 _postNorm 20 const 1e-1 & √
            # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 8 _postNorm 20 const 5e-2 & √
            # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 8 _postNorm 20 const 0.0  & √
            # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 8 _postNorm 5 cosine 1e-1 & √
            # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 8 _postNorm 5 cosine 5e-2 & √
            # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 8 _postNorm 5 cosine 0.0  & √
            # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 8 _preNorm 20 const 1e-1 & √
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 8 _preNorm 20 const 5e-2 & √
            # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 8 _preNorm 20 const 0.0  & √
            # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 8 _preNorm 5 cosine 1e-1 & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 8 _preNorm 5 cosine 5e-2 & √
            # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 8 _preNorm 5 cosine 0.0  & √
        # √ layer=9
            # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 9 _postNorm 20 const 1e-1 & √
            # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 9 _postNorm 20 const 5e-2 & √
            # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 9 _postNorm 5 cosine 1e-1 & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 9 _postNorm 5 cosine 5e-2 & √
            # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 9 _preNorm  20 const 1e-1 & √
            # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 9 _preNorm  20 const 5e-2 & √
            # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 9 _preNorm  5 cosine 1e-1 & √
            # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 9 _preNorm  5 cosine 5e-2 & √
        # √ layer=11
            # gpu26: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 11 _postNorm 20 const 1e-1 & √
            # gpu26: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 11 _postNorm 20 const 5e-2 & √
            # gpu26: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 11 _postNorm 20 const 0.0  & √
            # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 11 _postNorm 5 cosine 1e-1 & √
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 11 _postNorm 5 cosine 5e-2 & √
            # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 11 _postNorm 5 cosine 0.0  & √
            # gpu26: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 11 _preNorm 20 const 1e-1 & √
            # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 11 _preNorm 20 const 5e-2 & √
            # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 11 _preNorm 20 const 0.0  & √
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 11 _preNorm 5 cosine 1e-1 & √
            # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 11 _preNorm 5 cosine 5e-2 & √
            # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 11 _preNorm 5 cosine 0.0  & √
        # √ layer=12
            # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 12 _postNorm 20 const 1e-1 & √
            # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 12 _postNorm 20 const 5e-2 & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 12 _postNorm 5 cosine 1e-1 & √
            # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 12 _postNorm 5 cosine 5e-2 & √
            # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 12 _preNorm  20 const 1e-1 & √
            # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 12 _preNorm  20 const 5e-2 & √
            # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 12 _preNorm  5 cosine 1e-1 & √
            # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 12 _preNorm  5 cosine 5e-2 & √
        # √ layer=14
            # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 14 _postNorm 20 const 1e-1 & √
            # gpu27: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 14 _postNorm 20 const 5e-2 & √
            # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 14 _postNorm 20 const 0.0  & √
            # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 14 _postNorm 5 cosine 1e-1 & √
            # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans.py 14 _postNorm 5 cosine 5e-2 & √
            # gpu41: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans.py 14 _postNorm 5 cosine 0.0  & √
            # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans.py 14 _preNorm 20 const 1e-1 & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans.py 14 _preNorm 20 const 5e-2 & √
            # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_trans.py 14 _preNorm 20 const 0.0  & √
            # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_trans.py 14 _preNorm 5 cosine 1e-1 & √
            # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_trans.py 14 _preNorm 5 cosine 5e-2 & √
            # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans.py 14 _preNorm 5 cosine 0.0  & √
