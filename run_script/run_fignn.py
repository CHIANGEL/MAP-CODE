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
model = 'FiGNN'
num_hidden_layers = 10
dropout = '0.0'
res_conn = '_ResConn'
reuse_graph_layer = '_ReuseGraph'

# Run the train process
seed = sys.argv[2]
weight_decay = sys.argv[3]
num_hidden_layers = sys.argv[4]
for weight_decay in ['1e-1', '5e-2']:
    for seed in [42, 43, 44]:
        subprocess.run(['python', 'train.py', 
                        '--res_conn' if res_conn == '_ResConn' else '', 
                        '--reuse_graph_layer' if reuse_graph_layer == '_ReuseGraph' else '', 
                        f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                        f'hl{num_hidden_layers}_drop{dropout}{res_conn}{reuse_graph_layer}',

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
                        f'--hidden_dropout_rate={dropout}',
                        ])

# √ criteo_x4_core10_emb32 | wandb seed weight_decay num_layer
    # for weight_decay in ['1e-1', '5e-2']:
    # for seed in [42, 43, 44]:
    # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fignn.py criteo_x4_core10_emb32 seed wd 3 & √
    # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn.py criteo_x4_core10_emb32 seed wd 6 & √
    # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn.py criteo_x4_core10_emb32 seed wd 9 & √

# avazu_x4_core2_emb64
    # √ 基于最优结构，eb=64，wd=1e-num_hidden_layers=2，重复实验 | seed epoch lr_sched
        # for seed in [42, 43, 44, 45, 46]:
        # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn.py seed 20 const & √

    # √ eb=32
        # gpu17: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 1e-1 2 & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 1e-1 3 & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 python -u run_fignn.py 1e-1 4 & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 python -u run_fignn.py 1e-1 5 & √
        # gpu27: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 1e-1 6 & √
        # gpu27: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 1e-1 7 & √
        # gpu27: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 1e-1 8 & √
        # gpu27: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 1e-1 9 & √
        # gpu27: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 1e-1 10 & √
        # gpu27: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 1e-1 12 & √
        # gpu27: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 1e-1 13 & √
        # gpu27: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 1e-1 14 & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 5e-2 2 & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 5e-2 3 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 5e-2 4 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 5e-2 5 & √
        # gpu28: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 5e-2 6 & √
        # gpu27: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 5e-2 7 & √
        # gpu28: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 5e-2 8 & √
        # gpu28: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 5e-2 9 & √
        # gpu28: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 5e-2 10 & √
        # gpu28: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 5e-2 12 & √
        # gpu28: CUDA_VISIBLE_DEVICES=0 python -u run_fignn.py 5e-2 13 & √
        # gpu27: CUDA_VISIBLE_DEVICES=1 python -u run_fignn.py 5e-2 14 & √

    # √ eb=64
        # for num_hidden_layers in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]:
        # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn.py 1e-1 & √
