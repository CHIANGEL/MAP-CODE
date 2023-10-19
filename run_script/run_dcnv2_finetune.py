import os, sys
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
model = 'DCNv2'
hidden_size = sys.argv[2]
num_hidden_layers = sys.argv[3]
num_cross_layers = sys.argv[4]
dropout = '0.0'

# Finetune model path
ptEpoch = sys.argv[5]
ptType = sys.argv[6]
sampling_method = sys.argv[7]
mask_ratio = sys.argv[8]
ptDrop = '0.0'
ptBS = utils.get_ptBS(ptType, model, dataset_name)
ptWD = '5e-2'
ptNeg = sys.argv[18]
ptProj = 32
ptAeps, ptBetas = ('1e-8', '0.9,0.999')
ptGumbel = sys.argv[9]
ptGw = sys.argv[10]
ptDw = sys.argv[11]
RFD_G = sys.argv[12]
load_step = utils.get_load_step(dataset_name, ptEpoch, ptBS)
finetune_model_path = f'../{wandb_name}/{model}/pretrain/{ptType}/{split_name}_{model}_core{n_core}_epoch{ptEpoch}_bs{ptBS}_nceLossperWord_lr1e-3_cosineSched_wd{ptWD}_Aeps{ptAeps}_Abeta{ptBetas}_warmup{warmup}_1GPU_' + \
f'hs{hidden_size}_hl{num_hidden_layers}_cl{num_cross_layers}_drop{dropout}_' + \
f'{ptType}_{sampling_method}Sample_mask{mask_ratio}_neg{ptNeg}_proj{ptProj}_Gumbel{ptGumbel}_mfpW{ptGw}_rfdW{ptDw}_rfdg{RFD_G}/{load_step}.model'
assert os.path.exists(finetune_model_path), finetune_model_path

# Run the process
seed = sys.argv[13]
weight_decay = sys.argv[14]
epoch = sys.argv[15]
lr_sched = sys.argv[16]
lr = sys.argv[17]
lrs = ['1e-3', '7e-4', '5e-4']
for lr in lrs:
    for weight_decay in ['5e-2', '1e-1']:
        for epoch, lr_sched in [(1, 'cosine'), (2, 'cosine'), (3, 'cosine')]:
            for seed in [42]:
                subprocess.run(['python', 'train.py', '--finetune',
                                f'--output_dir=../{wandb_name}/{model}/finetune/{ptType}/{seed}/{split_name}_{model}_pt[{ptType}-{ptEpoch}-{ptBS}-{ptNeg}-{ptProj}-{sampling_method}-{mask_ratio}-{ptGumbel}-{ptGw}-{ptDw}-{RFD_G}]_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                                f'hs{hidden_size}_hl{num_hidden_layers}_cl{num_cross_layers}_drop{dropout}',
                                f'--finetune_model_path={finetune_model_path}',

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
