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
model = 'xDeepFM'
use_lr = 'NoLr'
hidden_size = 1000
num_hidden_layers = 0
cin_layer_units = '50,50'
dropout = '0.0'

# Run the train process
seed = sys.argv[2]
weight_decay = sys.argv[3]
hidden_size = sys.argv[4]
num_hidden_layers = sys.argv[5]
cin_layer_units = sys.argv[6]
for weight_decay in ['1e-1', '5e-2']:
    for seed in [42, 43, 44]:
        subprocess.run(['python', 'train.py',
                        '--use_lr' if use_lr == 'Lr' else '', 
                        f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                        f'cin{cin_layer_units}_hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}_{use_lr}',
                        
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
                        f'--cin_layer_units={cin_layer_units}',
                        f'--hidden_dropout_rate={dropout}',
                        ])

# √ criteo_x4_core10_emb16 | wandb seed wd hidden_size num_hidden_layers cin_layer_unit
    # √ CIN
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 12,12 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 12,12,12 & √
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 12,12,12,12 & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 12,12,12,12,12 & √
        # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 12,12,12,12,12,12 & √
        # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 12,12,12,12,12,12,12 & √
        # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 25,25 & √
        # gpu41: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 25,25,25 & √
        # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 25,25,25,25 & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 25,25,25,25,25 & √
        # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 25,25,25,25,25,25 & √
        # gpu05: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 25,25,25,25,25,25,25 & √
        # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 50,50 & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 0 50,50,50 & √
    # √ xDeepFM
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 3 25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 3 25,25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 6 25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 6 25,25,25,25,25 & √
        # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 9 25,25,25,25 & √
        # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 9 25,25,25,25,25 & √
        # gpu: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 12 25,25,25,25 &
        # gpu: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 1000 12 25,25,25,25,25 &
        # gpu10: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 3 25,25,25,25 & √
        # gpu10: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 3 25,25,25,25,25 & √
        # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 6 25,25,25,25 & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 6 25,25,25,25,25 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 9 25,25,25,25 & √
        # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 9 25,25,25,25,25 & √
        # gpu: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 12 25,25,25,25 &
        # gpu: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb16 seed wd 2000 12 25,25,25,25,25 &

# √ criteo_x4_core10_emb32 | seed wd epoch lr_sched cin_layer_unit
    # √ CIN
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu22: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 12,12 & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 12,12,12 & √
        # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 12,12,12,12 & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 12,12,12,12,12 & √
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 25,25,25 & √
        # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 25,25,25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 25,25,25,25,25 & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 20 const 50,50 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 12,12 & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 12,12,12 & √
        # gpu41: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 12,12,12,12 & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 12,12,12,12,12 & √
        # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 25,25,25 & √
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 25,25,25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 25,25,25,25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py seed wd 5 cosine 50,50 & √
    # √ xDeepFM
        # for weight_decay in ['1e-1', '5e-2']:
        # for seed in [42, 43, 44]:
        # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 3 25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 3 25,25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 6 25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 6 25,25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 9 25,25,25,25 & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 9 25,25,25,25,25 & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 12 25,25,25,25 & √
        # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 1000 12 25,25,25,25,25 & √
        # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 3 25,25,25,25 & √
        # gpu26: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 3 25,25,25,25,25 & √
        # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 6 25,25,25,25 & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 6 25,25,25,25,25 & √
        # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 9 25,25,25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 9 25,25,25,25,25 & √
        # gpu39: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 12 25,25,25,25 & √
        # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py criteo_x4_core10_emb32 seed wd 2000 12 25,25,25,25,25 & √

# √ avazu_x4_core2_emb64 | wandb, seed, wd, hidden_size, hidden_layer, cin_layer_units
    # for weight_decay in ['1e-1', '5e-2']:
    # for seed in [42, 43, 44]:
    # √ CIN调参
        # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 12,12 & √
        # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 12,12,12 & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 12,12,12,12 & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 12,12,12,12,12 & √
        # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 25,25 & √
        # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 25,25,25 & √
        # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 25,25,25,25 & √
        # gpu05: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 25,25,25,25,25 & √
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 0 50,50 & √
    # √ xDeepFM调参
        # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000  3 25,25 & √
        # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000  6 25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000  9 25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 1000 12 25,25 & √
        # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 2000  3 25,25 & √
        # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 2000  6 25,25 & √
        # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 2000  9 25,25 & √
        # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_xdeepfm.py avazu_x4_core2_emb64 seed wd 2000 12 25,25 & √
