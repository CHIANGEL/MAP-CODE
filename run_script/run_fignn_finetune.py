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
batch_size = 2048
lr = '1e-3'
weight_decay = '1e-1'
lr_sched = 'const'
adam_epsilon, adam_betas = ('1e-8', '0.9,0.999')
warmup = '0.0'

# Model config
model = 'FiGNN'
num_hidden_layers = sys.argv[2]
dropout = '0.0'
res_conn = '_ResConn'
reuse_graph_layer = '_ReuseGraph'

# Finetune model path
ptEpoch = sys.argv[3]
ptType = sys.argv[4]
sampling_method = sys.argv[5]
mask_ratio = sys.argv[6]
ptDrop = '0.0'
ptBS = utils.get_ptBS(ptType, model, dataset_name)
ptWD = '5e-2'
ptNeg = 25
ptProj = 32
ptAeps, ptBetas = ('1e-8', '0.9,0.999')
ptGumbel = sys.argv[7]
ptGw = sys.argv[8]
ptDw = sys.argv[9]
RFD_G = sys.argv[10]
load_step = utils.get_load_step(dataset_name, ptEpoch, ptBS)
finetune_model_path = f'../{wandb_name}/{model}/pretrain/{ptType}/{split_name}_{model}_core{n_core}_epoch{ptEpoch}_bs{ptBS}_nceLossperWord_lr1e-3_cosineSched_wd{ptWD}_Aeps{ptAeps}_Abeta{ptBetas}_warmup{warmup}_1GPU_' + \
f'hl{num_hidden_layers}_drop{dropout}{res_conn}{reuse_graph_layer}_' + \
f'{ptType}_{sampling_method}Sample_mask{mask_ratio}_neg{ptNeg}_proj{ptProj}_Gumbel{ptGumbel}_mfpW{ptGw}_rfdW{ptDw}_rfdg{RFD_G}/{load_step}.model'
assert os.path.exists(finetune_model_path), finetune_model_path

# Run the process
seed = sys.argv[8]
weight_decay = sys.argv[9]
epoch = sys.argv[10]
lr_sched = sys.argv[11]
lr = sys.argv[12]
for lr in ['1e-3', '7e-4', '5e-4']:
    for weight_decay in ['5e-2', '1e-1']:
        for epoch, lr_sched in [(1, 'cosine'), (2, 'cosine'), (3, 'cosine')]:
            for seed in [42]:
                subprocess.run(['python', 'train.py', '--finetune',
                                '--res_conn' if res_conn == '_ResConn' else '', 
                                '--reuse_graph_layer' if reuse_graph_layer == '_ReuseGraph' else '', 
                                f'--output_dir=../{wandb_name}/{model}/finetune/{ptType}/{seed}/{split_name}_{model}_pt[{ptType}-{ptEpoch}-{ptBS}-{ptNeg}-{ptProj}-{sampling_method}-{mask_ratio}-{ptGumbel}-{ptGw}-{ptDw}-{RFD_G}]_core{n_core}_epoch{epoch}_bs{batch_size}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_' + 
                                f'hl{num_hidden_layers}_drop{dropout}{res_conn}{reuse_graph_layer}',
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
                                f'--hidden_size={embed_size}',
                                f'--num_hidden_layers={num_hidden_layers}',
                                f'--hidden_dropout_rate={dropout}',
                                ])
                        
# Masked Feature Prediction
    # √ avazu_x4_core2_emb64 | 2 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 MFP normal 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 MFP normal 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 MFP normal 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
    # √ criteo_x4_core10_emb32 | 9 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 MFP normal 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 MFP normal 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 MFP normal 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √

            
# Replaced Feature Detection
    # √ avazu_x4_core2_emb64 | 2 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 RFD randint 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu27: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 RFD randint 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu27: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 RFD randint 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
    # √ criteo_x4_core10_emb32 | 9 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 RFD randint 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 RFD randint 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 RFD randint 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √

            
# SCARF: Contrastive Learning
    # √ avazu_x4_core2_emb64 | 2 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 SCARF randint 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 SCARF randint 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu41: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb64 2 60 SCARF randint 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
    # √ criteo_x4_core10_emb32 | 9 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu32: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 SCARF randint 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 SCARF randint 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 60 SCARF randint 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √


# MF4UIP
    # √ avazu_x4_core2_emb32 | 2 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb32 2 5 MF4UIP normal 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu32: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb32 2 5 MF4UIP normal 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu32: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py avazu_x4_core2_emb32 2 5 MF4UIP normal 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
    # √ criteo_x4_core10_emb32 | 9 | wandb, hidden_layer, ptEpoch, ptType, sample_method, mask_ratio, ptGumbel, ptGw, ptDw, RFD_G, seed, weight_decay, epoch, lr_sched, lr
        # √ ptEpoch=60
            # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 5 MF4UIP normal 0.1 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu28: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 5 MF4UIP normal 0.2 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_fignn_finetune.py criteo_x4_core10_emb32 9 5 MF4UIP normal 0.3 1.0 1.0 1.0 Unigram seed wd epoch sched lr & √
            