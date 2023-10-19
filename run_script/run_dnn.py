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
    data_dir = '../data/criteo/criteo_x4'
else:
    assert 0
seed = 42
split_name = sys.argv[2]
epoch = 20
batch_size = 4096
lr = '1e-3'
weight_decay = '1e-1'
lr_sched = 'const'
adam_epsilon, adam_betas = ('1e-8', '0.9,0.999')
warmup = '0.0'

# Model config
model = 'DNN'
hidden_size = 1000
num_hidden_layers = 3
dropout = '0.0'

# Run the train process
seed = sys.argv[3]
weight_decay = sys.argv[4]
num_hidden_layers = sys.argv[5]
hidden_size = sys.argv[6]
for weight_decay in ['1e-1', '5e-2']:
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

# criteo_x4_core10_emb16
    # √ split_name='x4-full' | seed wd hidden_layer hidden_size epoch lr_sched
        # for num_hidden_layers in [12, 9, 6, 3]:
        # for seed in [42, 43, 44]:
        # gpu10: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 1000 20 const & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 2000 20 const & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 1000 20 const & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 2000 20 const & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 1000 5 cosine & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 2000 5 cosine & √
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 1000 5 cosine & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 2000 5 cosine & √
    # split_name='pt' | wandb, split, seed, wd, hidden_layer, hidden_size
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 3  1000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 6  1000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 9  1000 & √
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 12 1000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 3  2000 & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 6  2000 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 9  2000 & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 pt seed wd 12 2000 & √
    # split_name='ch' | wandb, split, seed, wd, hidden_layer, hidden_size
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42]:
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 3  1000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 6  1000 & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 9  1000 & √
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 12 1000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 3  2000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 6  2000 & √
        # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 9  2000 & √
        # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_dnn.py criteo_x4_core10_emb16 ch seed wd 12 2000 & √

# √ criteo_x4_core10_emb32
    # for num_hidden_layers in [12, 9, 6, 3]:
    # for seed in [42, 43, 44]:
    # 调DNN结构 | seed wd hidden_layer hidden_size epoch lr_sched
        # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 1000 20 const & √
        # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 2000 20 const & √
        # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 1000 20 const & √
        # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 2000 20 const & √
        # gpu41: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 1000 5 cosine & √
        # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 hidden_layer 2000 5 cosine & √
        # gpu41: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 1000 5 cosine & √
        # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 hidden_layer 2000 5 cosine & √

# √ criteo_x4_core10_emb64
    # 调DNN结构 | seed wd hidden_layer hidden_size
        # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 3  1000 & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 6  1000 & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 9  1000 & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 12 1000 & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 3  2000 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 6  2000 & √
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 9  2000 & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 12 2000 & √
        # gpu22: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 3  1000 & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 6  1000 & √
        # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 9  1000 & √
        # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 12 1000 & √
        # gpu10: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 3  2000 & √
        # gpu10: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 6  2000 & √
        # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 9  2000 & √
        # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 12 2000 & √

# √ avazu_x4_core2_emb64
    # √ 调DNN结构 | seed wd hidden_layer hidden_size
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 3  1000 & √
        # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 6  1000 & √
        # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 9  1000 & √
        # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 12 1000 & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 3  2000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 6  2000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 9  2000 & √
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dnn.py seed 1e-1 12 2000 & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 3  1000 & √
        # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 6  1000 & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 9  1000 & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 12 1000 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 3  2000 & √
        # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 6  2000 & √
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 9  2000 & √
        # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dnn.py seed 5e-2 12 2000 & √