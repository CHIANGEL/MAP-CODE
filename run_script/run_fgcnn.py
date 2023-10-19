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
model = 'FGCNN'
hidden_size = 2000
num_hidden_layers = 0
dropout = '0.0'
hidden_act = 'relu'
channels = '14,16,18,20'
kernel_heights = '7,7,7,7'
pooling_sizes = '2,2,2,2'
recombined_channels = '3,3,3,3'
conv_act = 'tanh'

# Run the train process
seed = sys.argv[2]
weight_decay = sys.argv[3]
epoch lr_sched = sys.argv[4], sys.argv[5]
channels = sys.argv[6]
for seed in [42, 43, 44]:
    subprocess.run(['python', 'train.py',
                    f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                    f'hs{hidden_size}_hl{num_hidden_layers}_channels{channels}_REchannels{recombined_channels}',

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
                    f'--hidden_act={hidden_act}',
                    f'--channels={channels}',
                    f'--kernel_heights={kernel_heights}',
                    f'--pooling_sizes={pooling_sizes}',
                    f'--recombined_channels={recombined_channels}',
                    f'--conv_act={conv_act}',
                    ])

# √ criteo_x4_core10_emb16
    # recombine='3,3,3,3' | wandb seed weight_decay epoch lr_sched channel
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 1e-1 5 cosine 8,10,12,14  & √
        # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 1e-1 5 cosine 14,16,18,20 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 1e-1 20 const 8,10,12,14  & √
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 1e-1 20 const 14,16,18,20 & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 5e-2 5 cosine 8,10,12,14  & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 5e-2 5 cosine 14,16,18,20 & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 5e-2 20 const 8,10,12,14  & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py criteo_x4_core10_emb16 seed 5e-2 20 const 14,16,18,20 & √

# √ criteo_x4_core10_emb32
    # recombine='3,3,3,3' | seed weight_decay epoch lr_sched channel
        # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py seed 1e-1 5 cosine 8,10,12,14  & √
        # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py seed 1e-1 5 cosine 14,16,18,20 & √
        # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py seed 1e-1 20 const 8,10,12,14  & √
        # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fgcnn.py seed 1e-1 20 const 14,16,18,20 & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py seed 5e-2 5 cosine 8,10,12,14  & √
        # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py seed 5e-2 5 cosine 14,16,18,20 & √
        # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py seed 5e-2 20 const 8,10,12,14  & √
        # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_fgcnn.py seed 5e-2 20 const 14,16,18,20 & √

# avazu_x4_core2_emb32
    # √ recombine='3,3,3,3' | seed channel epoch lr_sched
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py 42 8,10,12,14 5 cosine  & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py 42 14,16,18,20 5 cosine & √
        # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py 42 8,10,12,14 20 const  & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py 42 14,16,18,20 20 const & √
        # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py 43 8,10,12,14 5 cosine  & √
        # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py 43 14,16,18,20 5 cosine & √
        # gpu26: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py 43 8,10,12,14 20 const  & √
        # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fgcnn.py 43 14,16,18,20 20 const & √
        # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py 44 8,10,12,14 5 cosine  & √
        # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py 44 14,16,18,20 5 cosine & √
        # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py 44 8,10,12,14 20 const  & √
        # gpu19: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py 44 14,16,18,20 20 const & √
        # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py 45 8,10,12,14 5 cosine  & √
        # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py 45 14,16,18,20 5 cosine & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fgcnn.py 45 8,10,12,14 20 const  & √
        # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fgcnn.py 45 14,16,18,20 20 const & √
        # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py 46 8,10,12,14 5 cosine  & √
        # gpu29: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fgcnn.py 46 14,16,18,20 5 cosine & √
        # gpu29: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fgcnn.py 46 8,10,12,14 20 const  & √
        # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fgcnn.py 46 14,16,18,20 20 const & √
