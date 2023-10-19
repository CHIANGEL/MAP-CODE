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

num_hidden_layers = sys.argv[12]
norm_first = sys.argv[13]

if gpu_num == '1':
    proc_prefix_list = ['python', 'train.py']
else:
    # proc_prefix_list = ['torchrun', f'--nproc_per_node={gpu_num}', 'train.py']
    proc_prefix_list = ['python', '-m', 'torch.distributed.launch', f'--nproc_per_node={gpu_num}', 'train.py']
subprocess.run(proc_prefix_list + 
                ['--pretrain', '--batch_first', 
                '--norm_first' if norm_first == '_preNorm' else '',
                '--use_lr' if use_lr == 'Lr' else '', 
                f'--output_dir=../{wandb_name}/{model}/pretrain/{pt_type}/{split_name}_{model}_core{n_core}_epoch{epoch}_bs{batch_size}_{pt_loss}Loss{per_word}_lr{lr}_{lr_sched}Sched_wd{weight_decay}_Aeps{adam_epsilon}_Abeta{adam_betas}_warmup{warmup}_{gpu_num}GPU_' + \
                f'tl{num_hidden_layers}_th{num_attn_heads}_td{dropout}{norm_first}_FFN{intermediate_size}_' + \
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
                f'--hidden_size={embed_size}',
                f'--num_hidden_layers={num_hidden_layers}',
                f'--num_attn_heads={num_attn_heads}',
                f'--intermediate_size={intermediate_size}',
                f'--output_reduction={reduction}',
                f'--hidden_dropout_rate={dropout}',
                f'--dnn_size={dnn_size}',
                f'--num_dnn_layers={num_dnn_layers}',
                ])

# Masked Feature Prediction
    # √ avazu_x4_core2_emb64 | (3, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ ptEpoch=10
            # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 10 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 10 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 10 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ ptEpoch=20
            # gpu41: CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 20 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 20 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 20 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ ptEpoch=30
            # gpu14: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 30 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 30 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu41: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 30 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ ptEpoch=40
            # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 40 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 40 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 40 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ ptEpoch=50
            # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 50 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 50 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu26: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 50 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ ptEpoch=60
            # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu40: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
    # √ criteo_x4_core10_emb32 | (9, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ ptEpoch=10
            # gpu41: CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu41: CUDA_VISIBLE_DEVICES=6 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ ptEpoch=20
            # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 20 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 20 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu24: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 20 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ ptEpoch=30
            # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 30 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 30 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 30 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ ptEpoch=40
            # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 40 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu34: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 40 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 40 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ ptEpoch=50
            # gpu30: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 50 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 50 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 50 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ ptEpoch=60
            # gpu39: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 MFP normal 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu39: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 MFP normal 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu39: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 MFP normal 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √

# Replaced Feature Detection
    # √ avazu_x4_core2_emb64 | (3, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ Unigram, ptEpoch=10
            # gpu13: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu33: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ Unigram, ptEpoch=20
            # gpu40: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 20 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 20 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu32: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 20 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ Unigram, ptEpoch=30
            # gpu41: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 30 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu38: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 30 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 30 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ Unigram, ptEpoch=40
            # gpu04: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 40 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 40 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu40: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 40 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ Unigram, ptEpoch=50
            # gpu22: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 50 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu12: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 50 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 50 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
        # √ Unigram, ptEpoch=60
            # gpu22: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu35: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu34: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
    # √ criteo_x4_core10_emb32 | (9, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ Unigram, ptEpoch=10
            # √ (3, _postNorm)
                # gpu04: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _postNorm & √
                # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _postNorm & √
                # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _postNorm & √
            # √ (3, _preNorm)
                # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
                # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
                # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # √ (6, _postNorm)
                # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 6 _postNorm & √
                # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 6 _postNorm & √
                # gpu21: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 6 _postNorm & √
            # √ (6, _preNorm)
                # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 6 _preNorm & √
                # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 6 _preNorm & √
                # gpu32: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 6 _preNorm & √
            # √ (9, _postNorm)
                # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 9 _postNorm & √
                # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 9 _postNorm & √
                # gpu17: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 9 _postNorm & √
            # √ (9, _preNorm)
                # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
                # gpu13: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
                # gpu17: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # √ (12, _postNorm)
                # gpu12: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 12 _postNorm & √
                # gpu32: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 12 _postNorm & √
                # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 12 _postNorm & √
            # √ (12, _preNorm)
                # gpu32: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 12 _preNorm & √
                # gpu32: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 12 _preNorm & √
                # gpu32: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 10 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 12 _preNorm & √
        # √ Unigram, ptEpoch=20
            # gpu41: CUDA_VISIBLE_DEVICES=4 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 20 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu17: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 20 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu18: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 20 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ Unigram, ptEpoch=30
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 30 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 30 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu17: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 30 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ Unigram, ptEpoch=40
            # gpu18: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 40 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 40 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu18: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 40 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ Unigram, ptEpoch=50
            # gpu40: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 50 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu37: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 50 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu35: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 50 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
        # √ Unigram, ptEpoch=60
            # √ (3, _preNorm)
                # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
                # gpu37: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
                # gpu27: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # √ (6, _preNorm)
                # gpu21: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 6 _preNorm & √
                # gpu22: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 6 _preNorm & √
                # gpu33: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 6 _preNorm & √
            # √ (9, _preNorm)
                # gpu32: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
                # gpu40: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
                # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 4096 RFD randint 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √

# SCARF: Contrastive Learning
    # √ avazu_x4_core2_emb64 | (3, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ Unigram, ptEpoch=60
            # gpu18: CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 2048 SCARF randint 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu38: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 2048 SCARF randint 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu36: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 60 2048 SCARF randint 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
    # √ criteo_x4_core10_emb32 | (9, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ Unigram, ptEpoch=60
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 2048 SCARF randint 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu33: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 2048 SCARF randint 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu33: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 60 2048 SCARF randint 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √

# MF4UIP
    # √ avazu_x4_core2_emb64 | (3, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ Unigram, ptEpoch=60
            # gpu28: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 5 256 MF4UIP normal 0.1 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu27: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 5 256 MF4UIP normal 0.2 1.0 1.0 1.0 Unigram 3 _preNorm & √
            # gpu27: CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python -u run_trans_pretrain.py avazu_x4_core2_emb64 1 5 256 MF4UIP normal 0.3 1.0 1.0 1.0 Unigram 3 _preNorm & √
    # √ criteo_x4_core10_emb32 | (9, _preNorm) | wandb, gpu_num, epoch, bs, pt_type, sample_method, mask_ratio, gumbel, mfp_w, rfd_w, RFD_G, hidden_layer, norm_first
        # √ Unigram, ptEpoch=60
            # gpu22: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 5 256 MF4UIP normal 0.1 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu22: CUDA_VISIBLE_DEVICES=2 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 5 256 MF4UIP normal 0.2 1.0 1.0 1.0 Unigram 9 _preNorm & √
            # gpu21: CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python -u run_trans_pretrain.py criteo_x4_core10_emb32 1 5 256 MF4UIP normal 0.3 1.0 1.0 1.0 Unigram 9 _preNorm & √
