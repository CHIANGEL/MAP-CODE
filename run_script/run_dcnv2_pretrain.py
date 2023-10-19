import sys
import subprocess
import utils

# Training args
wandb_name = sys.argv[1]
dataset_name, n_core, embed_size = utils.parse_wandb_name(wandb_name)
data_dir = utils.get_data_dir(dataset_name)
split_name = 'x4-full'
epoch = 60
batch_size = 4096
lr = '1e-3'
weight_decay = '5e-2'
lr_sched = 'cosine'
adam_epsilon, adam_betas = ('1e-8', '0.9,0.999')
warmup = '0.0'
gpu_num = '1'

# Pretrain Args
pt_type = 'MFP'
sampling_method = 'normal'
mask_ratio = '0.1'
pt_loss = 'nce'
per_word = 'perWord'
pt_neg_num = 25
proj_size = 32
gumbel_temp = '1.0'
mfp_w = '1.0'
rfd_w = '1.0'
RFD_G = 'Model'

# Model config
model = 'DCNv2'
hidden_size = 1000
num_hidden_layers = 8
num_cross_layers = 8
dropout = '0.0'

# Run the process
gpu_num = sys.argv[2]
epoch = sys.argv[3]
batch_size = sys.argv[4]

pt_type = sys.argv[5]
sampling_method = sys.argv[6]
mask_ratio = sys.argv[7]
gumbel_temp = sys.argv[8]
mfp_w = sys.argv[9]
rfd_w = sys.argv[10]
RFD_G = sys.argv[11]

hidden_size = sys.argv[12]
num_hidden_layers = sys.argv[13]
num_cross_layers = sys.argv[14]

pt_neg_num = sys.argv[15]

if gpu_num == '1':
    proc_prefix_list = ['python', 'train.py']
else:
    # proc_prefix_list = ['torchrun', f'--nproc_per_node={gpu_num}', 'train.py']
    proc_prefix_list = ['python', '-m', 'torch.distributed.launch', f'--nproc_per_node={gpu_num}', 'train.py']
subprocess.run(proc_prefix_list + 
                ['--pretrain', 
                f'--output_dir=../{wandb_name}/{model}/pretrain/{pt_type}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_{pt_loss}Loss{per_word}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_{gpu_num}GPU_' + \
                f'hs{hidden_size}_hl{num_hidden_layers}_cl{num_cross_layers}_drop{dropout}_' + \
                f'{pt_type}_{sampling_method}Sample_mask{mask_ratio}_neg{pt_neg_num}_proj{proj_size}_Gumbel{gumbel_temp}_mfpW{mfp_w}_rfdW{rfd_w}_rfdg{RFD_G}',

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
                
                f'--pt_type={pt_type}',
                f'--sampling_method={sampling_method}',
                f'--mask_ratio={mask_ratio}',
                f'--pt_loss={pt_loss}',
                f'--pt_neg_num={pt_neg_num}',
                f'--proj_size={proj_size}',
                f'--gumbel_temp={gumbel_temp}',
                f'--G_w={mfp_w}',
                f'--D_w={rfd_w}',
                f'--RFD_G={RFD_G}',

                f'--model_name={model}',
                f'--embed_size={embed_size}',
                f'--hidden_size={hidden_size}',
                f'--num_hidden_layers={num_hidden_layers}',
                f'--num_cross_layers={num_cross_layers}',
                f'--hidden_dropout_rate={dropout}',
                ])
