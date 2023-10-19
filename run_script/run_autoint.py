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
batch_size = 4096
lr = '1e-3'
weight_decay = '1e-1'
lr_sched = 'const'
adam_epsilon, adam_betas = ('1e-8', '0.9,0.999')
warmup = '0.0'

# Model config
model = 'AutoInt'
num_attn_layers = 3
num_attn_heads = 1
attn_size = 64
attn_probs_dropout_rate = '0.0'
dnn_size = 1000
num_dnn_layers = 0
dnn_act = 'relu'
dnn_drop = '0.0'
res_conn = 'ResConn'
attn_scale = 'NoScale'
use_lr = 'NoLr'

# Run the train process
seed = sys.argv[2]
weight_decay = sys.argv[3]
num_attn_layers = sys.argv[4]
attn_size = sys.argv[5]
res_conn = 'ResConn'
attn_scale = 'NoScale'
for weight_decay in ['1e-1', '5e-2']:
    for seed in [42, 43, 44]:
        subprocess.run(['python', 'train.py', 
                        '--res_conn' if res_conn == 'ResConn' else '', 
                        '--attn_scale' if attn_scale == 'Scale' else '', 
                        '--use_lr' if use_lr == 'Lr' else '', 
                        f'--output_dir=../{wandb_name}/{model}/scratch/{seed}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                        f'attL{num_attn_layers}_attH{num_attn_heads}_attS{attn_size}_attD{attn_probs_dropout_rate}_{res_conn}_{attn_scale}_ds{dnn_size}_dl{num_dnn_layers}_{use_lr}',

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
                        f'--num_attn_layers={num_attn_layers}',
                        f'--num_attn_heads={num_attn_heads}',
                        f'--attn_size={attn_size}',
                        f'--attn_probs_dropout_rate={attn_probs_dropout_rate}',
                        f'--dnn_size={dnn_size}',
                        f'--num_dnn_layers={num_dnn_layers}',
                        f'--dnn_act={dnn_act}',
                        f'--dnn_drop={dnn_drop}',
                        ])
