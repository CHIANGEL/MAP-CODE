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
model = 'DCNv2'
hidden_size = 1000
num_hidden_layers = 0
num_cross_layers = 3
dropout = '0.0'

# Run the train process
seed = sys.argv[2]
weight_decay = sys.argv[3]
hidden_size = sys.argv[4]
num_hidden_layers = sys.argv[5]
num_cross_layers = sys.argv[6]
for hidden_size in [2000, 1000]:
    for weight_decay in ['1e-1', '5e-2']:
        for seed in [42, 43, 44]:
            subprocess.run(['python', 'train.py',
                            f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + \
                            f'hs{hidden_size}_hl{num_hidden_layers}_cl{num_cross_layers}_drop{dropout}',
                            
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
                            f'--num_cross_layers={num_cross_layers}',
                            f'--hidden_dropout_rate={dropout}',
                            ])

# criteo_x4_core10_emb16
    # split_name='x4-full'
        # √ 调CrossNet v2的结构，将num_hidden_layers置零 | wandb seed wd hidden_size num_hidden_layers num_cross_layers epoch lr_sched
            # for weight_decay in ['1e-1', '5e-2']:
            # for seed in [42, 43, 44]:
            # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 3  20 const & √
            # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 6  20 const & √
            # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 9  20 const & √
            # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 12 20 const & √
            # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 3  5 cosine & √
            # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 6  5 cosine & √
            # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 9  5 cosine & √
            # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd 1000 0 12 5 cosine & √
        # √ 调DCN v2的结构 | wandb seed wd hidden_size num_hidden_layers num_cross_layers
            # for hidden_size in [2000, 1000]:
            # for weight_decay in ['1e-1', '5e-2']:
            # for seed in [42, 43, 44]:
            # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 3  3  & √
            # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 3  6  & √
            # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 3  9  & √
            # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 3  12 & √
            # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 6  3  & √
            # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 6  6  & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 6  9  & √
            # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 6  12 & √
            # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 9  3  & √
            # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 9  6  & √
            # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 9  9  & √
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 9  12 & √
            # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 12 3  & √
            # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 12 6  & √
            # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 12 9  & √
            # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 seed wd hs 12 12 & √
    # split_name='pt'
        # 调CrossNet v2的结构，将num_hidden_layers置零 | wandb, split, seed, wd, num_hidden_layers, num_cross_layers
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu37: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 pt seed wd 0 3  & √
        # gpu37: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 pt seed wd 0 6  & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 pt seed wd 0 9  & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 pt seed wd 0 12 & √
    # split_name='ch'
        # 调CrossNet v2的结构，将num_hidden_layers置零 | wandb, split, seed, wd, num_hidden_layers, num_cross_layers
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42]:
        # gpu37: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 ch seed wd 0 3  & √
        # gpu37: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 ch seed wd 0 6  & √
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 ch seed wd 0 9  & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb16 ch seed wd 0 12 & √

# √ criteo_x4_core10_emb32
    # √ 调CrossNet v2的结构，将num_hidden_layers置零 | wandb seed wd num_hidden_layers num_cross_layers epoch lr_sched
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 3  20 const & √
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 6  20 const & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 9  20 const & √
        # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 12 20 const & √
        # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 15 20 const & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 18 20 const & √
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 3  5 cosine & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 6  5 cosine & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 9  5 cosine & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 12 5 cosine & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 15 5 cosine & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd 0 18 5 cosine & √
    # √ 调DCN v2的结构 | wandb seed wd hidden_size num_hidden_layers num_cross_layers
        # for hidden_size in [2000, 1000]:
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 3  3  & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 3  6  & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 3  9  & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 3  12 & √
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 6  3  & √
        # gpu26: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 6  6  & √
        # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 6  9  & √
        # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 6  12 & √
        # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 9  3  & √
        # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 9  6  & √
        # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 9  9  & √
        # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 9  12 & √
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 12 3  & √
        # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 12 6  & √
        # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 12 9  & √
        # gpu27: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py criteo_x4_core10_emb32 seed wd hs 12 12 & √

# √ criteo_x4_core10_emb64
    # 调CrossNet v2的结构，将num_hidden_layers置零 | seed wd num_hidden_layers num_cross_layers
        # for seed in [42, 43, 44]:
        # gpu37: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 3  & √
        # gpu39: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 6  & √
        # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 9  & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 12 & √
        # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 15 & √
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 18 & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 3  & √
        # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 6  & √
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 9  & √
        # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 12 & √
        # gpu37: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 15 & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 18 & √

# √ avazu_x4_core2_emb64
    # √ 调CrossNet v2的结构，将num_hidden_layers置零 |  seed wd num_hidden_layers num_cross_layers
        # for seed in [42, 43, 44, 45, 46]:
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 3  & √
        # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 6  & √
        # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 9  & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 12 & √
        # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 15 & √
        # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 18 & √
        # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py seed 5e-2 0 21 & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 3  & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 6  & √
        # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 9  & √
        # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 12 & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 15 & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 18 & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py seed 1e-1 0 21 & √

    # √ 基于最优结构，num_hidden_layers=6/8，num_cross_layers=8 | seed wd num_hidden_layers num_cross_layers
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 43 5e-2 6 8 & √
        # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 44 5e-2 6 8 & √
        # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 45 5e-2 6 8 & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 46 5e-2 6 8 & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 43 5e-2 8 8 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 44 5e-2 8 8 & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py 45 5e-2 8 8 & √
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py 46 5e-2 8 8 & √

    # √ seed=42 调最优结构 | seed wd num_hidden_layers num_cross_layers
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 2 2 & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 2 4 & √
        # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 2 6 & √
        # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 2 8 & √
        # gpu26: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 2 10 & √
        # gpu26: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 4 2 & √
        # gpu26: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 4 4 & √
        # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 4 6 & √
        # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 4 8 & √
        # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 4 10 & √
        # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 6 2 & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 6 4 & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 6 6 & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 6 8 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 6 10 & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 8 2 & √
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 8 4 & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 8 6 & √
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 8 8 & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 8 10 & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 10 2 & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 10 4 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 10 6 & √
        # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 10 8 & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_dcnv2.py 42 5e-2 10 10 & √