import logging
import os, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict
import pandas as pd
from tqdm import tqdm, trange
from sklearn.metrics import log_loss, roc_auc_score
import time
import gc

from config import Config
from models import BaseModel
from adv_attacks import get_adv_attack_model
from optimization import AdamW, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from training_args import TrainingArguments


try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn('W&B installed but not logged in. Run `wandb login` or set the WANDB_APT_KEY env variable.')
    else:
        _has_wandb = False if os.getenv('WANDB_DISABLED') else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: BaseModel, model_config: Config, training_args: TrainingArguments,
                 train_dataset: Dataset, eval_dataset: Dataset, data_collator):
        self.model = model
        self.model_config = model_config
        self.args = training_args
        self.device = self.args.device
        self.n_gpu = self.args.n_gpu
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        self.global_step = 0
        self.eval_metrics = []
        logger.info(f'setting device {self.device}')
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.best_eval_auc = 0
        self.best_eval_step = -1

        if self.args.local_rank in [-1, 0]:
            if is_wandb_available():
                self._steup_wandb()
            else:
                logger.info(
                    "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                    "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
                )

    def _steup_wandb(self):
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        run_name = os.path.split(self.args.output_dir)[-1]
        if self.args.pretrain:
            run_name = 'pre_' + run_name
        elif self.args.finetune:
            run_name = 'ft_' + run_name
        else:
            run_name = 'scr_' + run_name
        wandb.init(name=run_name, project=self.args.wandb_name,
                   config={**vars(self.args), **self.model_config.to_dict()})

    def get_dataloader(self, dataset, is_training=True):
        collator = self.data_collator
        batch_size = self.args.per_gpu_train_batch_size if is_training else self.args.per_gpu_eval_batch_size
        if self.n_gpu > 1 and is_training:
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    collate_fn=collator.collate_batch, 
                                    sampler=sampler)
        else:
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=is_training, 
                                    collate_fn=collator.collate_batch)
        return dataloader

    def get_optimizer(self, num_training_steps: int, num_warmup_steps: int):
        no_decay = ['bias', 'LayerNorm.weight']
        named_params = [(k, v) for k, v in self.model.named_parameters()]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        beta1, beta2 = self.args.adam_betas.split(',')
        beta1, beta2 = float(beta1), float(beta2)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon,
                          betas=(beta1, beta2))
        if self.args.lr_sched.lower() == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)
        elif self.args.lr_sched.lower() == 'const':
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
        else:
            raise NotImplementedError

        return optimizer, scheduler

    def train(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        # TODO add cosine scheduler
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info('***** running training *****')
        logger.info(f'  dataset_name = {self.args.dataset_name}')
        logger.info(f'  input_size = {self.model_config.input_size}')
        logger.info(f'  num_fields = {self.model_config.num_fields}')
        logger.info(f'  num_examples = {len(self.train_dataset)}')
        logger.info(f'  num_epochs = {self.args.num_train_epochs}')
        logger.info(f'  batch_size = {self.args.train_batch_size}')
        logger.info(f'  n_core = {self.args.n_core}')
        logger.info(f'  total_steps = {t_total}')
        logger.info(f'  warmup_steps = {t_warmup}')
        logger.info(f'  learning_rate = {self.args.learning_rate}')
        logger.info(f'  weight_decay = {self.args.weight_decay}')
        logger.info(f'  lr_sched = {self.args.lr_sched}')
        self.model.validate_model_config()

        self._patience = 0
        self._stop_training = False
        self.global_step = 0
        self.eval_metrics = []
        tr_loss, logging_loss = 0., 0.
        tr_labels, tr_probs = [], []
        total_adv_loss = 0.

        self.model.to(self.device)
        self.model.zero_grad()
        # wandb.watch(self.model, log='gradients', log_freq=10, log_graph=False)
        if self.model_config.adv_train:
            attack_model = get_adv_attack_model(self.model, self.model_config)

        with trange(self.args.num_train_epochs, desc='epoch', ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f'-------------------- epoch-{epoch} --------------------')
                self.model.train()
                for step, inputs in enumerate(self.train_dataloader):
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    loss.backward()
                    step_loss = loss.item()
                    if self.model_config.output_dim == 1:
                        tr_probs.extend(torch.sigmoid(outputs[1].detach().cpu()).numpy())
                    elif self.model_config.output_dim == 2:
                        tr_probs.extend(torch.softmax(outputs[1].detach().cpu(), dim=-1).numpy()[:, 1])
                    tr_labels.extend(inputs['labels'].detach().cpu().numpy())
                    pbar.set_description(f'epoch-{epoch}, loss={step_loss:.4f}')

                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # Adversarial training
                    if self.model_config.adv_train:
                        attack_model.attack(self.model_config.adv_epsilon) # perturb the embedding table
                        # optimizer.zero_grad() # zero_grad means we do not want the gradient of original input
                        adv_outputs = self.model(**inputs)
                        adv_loss = adv_outputs[0]
                        adv_loss.backward()
                        total_adv_loss += adv_loss.item()
                        attack_model.restore() # NOTE: restore the orignal embeddings before updating them

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    tr_loss += step_loss
                    if self.global_step % self.args.logging_steps == 0:
                        window_auc = roc_auc_score(np.int32(tr_labels), np.array(tr_probs))
                        _log = {'window_auc': window_auc,
                                'window_loss': (tr_loss - logging_loss) / self.args.logging_steps}
                        if self.model_config.adv_train:
                            _log['adv_loss'] = total_adv_loss / len(tr_labels)
                        tr_labels, tr_probs = [], []
                        if is_wandb_available() and self.args.local_rank:
                            wandb.log(_log, step=self.global_step)
                            # if self.model.model_name.lower() == 'punv3' and self.model_config.attn_agg:
                            #     attn_weights = self.model.attn_weights.detach().cpu().numpy()
                            #     print(attn_weights.shape)
                            #     print(attn_weights)
                            #     exit(0)
                        logger.info(f'step = {self.global_step}, training_loss = {_log["window_loss"]}, '
                                    f'training_auc = {window_auc}')
                        logging_loss = tr_loss
                        total_adv_loss = 0.

                self.eval()
                if self._stop_training:
                    break

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=['auc', 'log_loss'])))

    def eval(self, eval_dataset=None, test_eval=False):
        if eval_dataset is None:
            eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        else:
            eval_dataloader = self.get_dataloader(eval_dataset, is_training=False)
        if test_eval:
            logger.info(f'***** running TEST *****')
        else:
            logger.info(f'***** running eval *****')
        logger.info(f'  num examples = {len(eval_dataloader.dataset)}')
        logger.info(f'  batch size = {self.args.eval_batch_size}')
        eval_losses, eval_size, preds, probs, label_ids = 0, 0, [], [], []
        self.model.eval()

        for inputs in eval_dataloader:
            with torch.no_grad():
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                step_eval_loss, logits = outputs[:2]
                eval_losses += step_eval_loss.mean().item() * logits.size()[0]
                eval_size += logits.size()[0]
            if self.model_config.output_dim == 1:
                preds.extend(logits.detach().cpu().numpy())
                probs.extend(torch.sigmoid(logits.detach().cpu()).numpy())
            elif self.model_config.output_dim == 2:
                preds.extend(logits.detach().cpu().numpy()[:, 1])
                probs.extend(torch.softmax(logits.detach().cpu(), dim=-1).numpy()[:, 1])
            label_ids.extend(inputs['labels'].detach().cpu().numpy())

        preds, probs, label_ids = np.array(preds).astype("float64"), np.array(probs).astype("float64"), np.array(label_ids)
        auc = roc_auc_score(y_true=label_ids, y_score=probs)
        ll = log_loss(y_true=label_ids, y_pred=probs)
        
        self.eval_metrics.append([auc, ll])
        _log = {'learning_rate': self.scheduler.get_last_lr()[0], 'eval_auc': auc, 'eval_loss': ll,
                'avg_logits': preds.mean(), 'avg_probs': probs.mean()}
        logger.info(str(_log))
        if test_eval == False:
            if is_wandb_available() and self.args.local_rank:
                wandb.log(_log, step=self.global_step)
            if auc > self.best_eval_auc:
                self.best_eval_auc = auc
                self.best_eval_step = self.global_step
                self._patience = 0
                self.save_model(self.args.output_dir)
            else:
                self._patience += 1
            if self._patience > self.args.patience:
                self._stop_training = True

    def dynamic_mask(self, inputs, sampling_method='normal'):
        batch_size = inputs['input_ids'].shape[0]
        num_fields = self.model_config.num_fields
        mask_num = int(num_fields * self.args.mask_ratio)
        if sampling_method == 'normal':
            masked_index = torch.stack([torch.randperm(num_fields)[:mask_num] for i in range(batch_size)], dim=0)
        elif sampling_method == 'randint':
            masked_index = torch.randint(0, num_fields, (batch_size, mask_num), device=inputs['input_ids'].device)
        else:
            raise NotImplementedError

        if self.args.pt_type == 'MFP':
            mask_ids = torch.zeros(masked_index.size(), dtype=torch.int64, device=inputs['input_ids'].device).fill_(3)
            inputs['labels'] = torch.gather(inputs['input_ids'], 1, masked_index)
            inputs['input_ids'] = torch.scatter(inputs['input_ids'], 1, masked_index, mask_ids)
            inputs['masked_index'] = masked_index
        elif self.args.pt_type == 'RFD':
            if self.args.RFD_G == 'Unigram':
                sample_index = torch.randint(0, len(self.train_dataset), (batch_size * mask_num, ))
                replace_sample = torch.from_numpy(self.train_dataset.X[sample_index]).to(inputs['input_ids'].device)
                replace_feat = torch.gather(replace_sample, 1, masked_index.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs['input_ids']
                inputs['input_ids'] = torch.scatter(inputs['input_ids'], 1, masked_index, replace_feat)
                inputs['labels'] = (origin_input_ids != inputs['input_ids']).float()
            elif self.args.RFD_G == 'Uniform':
                replace_sample = torch.stack([torch.randint(self.model_config.idx_low[i], self.model_config.idx_high[i], (batch_size * mask_num, )) for i in range(num_fields)], dim=1)
                replace_feat = torch.gather(replace_sample, 1, masked_index.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs['input_ids']
                inputs['input_ids'] = torch.scatter(inputs['input_ids'], 1, masked_index, replace_feat)
                inputs['labels'] = (origin_input_ids != inputs['input_ids']).float()
            elif self.args.RFD_G == 'Whole-Uniform':
                replace_sample = torch.randint(10, self.model_config.input_size, (batch_size * mask_num, num_fields))
                replace_feat = torch.gather(replace_sample, 1, masked_index.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs['input_ids']
                inputs['input_ids'] = torch.scatter(inputs['input_ids'], 1, masked_index, replace_feat)
                inputs['labels'] = (origin_input_ids != inputs['input_ids']).float()
            elif self.args.RFD_G == 'Whole-Unigram':
                sample_index_0 = torch.randint(0, len(self.train_dataset), (batch_size * mask_num, ))
                replace_sample = torch.from_numpy(self.train_dataset.X[sample_index_0]).to(inputs['input_ids'].device)
                sample_index_1 = torch.randint(0, num_fields, (batch_size, mask_num), device=inputs['input_ids'].device)
                replace_feat = torch.gather(replace_sample, 1, sample_index_1.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs['input_ids']
                inputs['input_ids'] = torch.scatter(inputs['input_ids'], 1, masked_index, replace_feat)
                inputs['labels'] = (origin_input_ids != inputs['input_ids']).float()
            elif self.args.RFD_G == 'Model':
                mask_ids = torch.zeros(masked_index.size(), dtype=torch.int64, device=inputs['input_ids'].device).fill_(3)
                inputs['origin_input_ids'] = inputs['input_ids']
                inputs['labels'] = torch.gather(inputs['input_ids'], 1, masked_index)
                inputs['input_ids'] = torch.scatter(inputs['input_ids'], 1, masked_index, mask_ids)
                inputs['masked_index'] = masked_index
            else:
                raise NotImplementedError
        elif self.args.pt_type == 'SCARF':
            sample_index = torch.randint(0, len(self.train_dataset), (batch_size * mask_num, ))
            replace_sample = torch.from_numpy(self.train_dataset.X[sample_index]).to(inputs['input_ids'].device)
            replace_feat = torch.gather(replace_sample, 1, masked_index.view(-1, 1)).view(batch_size, mask_num)
            corrupt_view = torch.scatter(inputs['input_ids'], 1, masked_index, replace_feat)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], corrupt_view], dim=0)
            del inputs['labels']
        elif self.args.pt_type == 'MF4UIP':
            mask_ids = torch.zeros(masked_index.size(), dtype=torch.int64, device=inputs['input_ids'].device).fill_(3)
            labels = inputs['input_ids'] - self.model_config.idx_low.view(1, -1)
            inputs['labels'] = labels
            # inputs['labels'] = torch.gather(labels, 1, masked_index)
            inputs['input_ids'] = torch.scatter(inputs['input_ids'], 1, masked_index, mask_ids)
            inputs['masked_index'] = masked_index
        else:
            raise NotImplementedError

        return inputs

    def MFP_pretrain(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        # TODO add cosine scheduler
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info('***** running pretraining *****')
        logger.info(f'  dataset_name = {self.args.dataset_name}')
        logger.info(f'  input_size = {self.model_config.input_size}')
        logger.info(f'  num_fields = {self.model_config.num_fields}')
        logger.info(f'  num_examples = {len(self.train_dataset)}')
        logger.info(f'  num_epochs = {self.args.num_train_epochs}')
        logger.info(f'  batch_size = {self.args.train_batch_size}')
        logger.info(f'  n_gpu = {self.args.n_gpu}')
        logger.info(f'  per_gpu_train_batch_size = {self.args.per_gpu_train_batch_size}')
        logger.info(f'  total_steps = {t_total}')
        logger.info(f'  warmup_steps = {t_warmup}')
        logger.info(f'  learning_rate = {self.args.learning_rate}')
        logger.info(f'  weight_decay = {self.args.weight_decay}')
        logger.info(f'  lr_sched = {self.args.lr_sched}')
        logger.info(f'  mask_ratio = {self.args.mask_ratio}')
        logger.info(f'  pt_loss = {self.model_config.pt_loss}')
        logger.info(f'  pt_neg_num = {self.model_config.pt_neg_num}')
        logger.info(f'  share_neg = {self.model_config.share_neg}')
        logger.info(f'  pt_type = {self.model_config.pt_type}')
        self.model.validate_model_config()

        self.global_step = 0
        self.eval_metrics = []
        tr_loss, logging_loss = 0., 0.
        tr_acc, logging_acc = 0., 0.

        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model.zero_grad()

        start_time = time.time()
        # mask_time, todevice_time, model_time, back_time, metric_time, optim_time = 0, 0, 0, 0, 0, 0
        # self.pre_index_time, self.index_time, self.post_index_time = 0, 0, 0
        with trange(self.args.num_train_epochs, desc='epoch', ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f'-------------------- epoch-{epoch} --------------------')
                self.model.train()
                for step, inputs in enumerate(self.train_dataloader):
                    # t0 = time.time()

                    # assert inputs['input_ids'].shape[0] == self.args.train_batch_size
                    # continue
                    
                    # for method in ['normal', 'randint', 'alias']:
                    #     t0 = time.time()
                    #     iters = 50
                    #     for _ in range(iters):
                    #         inputs = self.dynamic_mask(inputs, method)
                    #     mask_time = (time.time() - t0) / iters
                    #     print(f'{method}: {mask_time:.6f}')
                    # assert 0

                    # print('==========================', inputs['input_ids'].shape)
                    # assert 0

                    inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                    # mask_time += (time.time() - t0)
                    # t0 = time.time()

                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    # todevice_time += (time.time() - t0)
                    # t0 = time.time()

                    outputs = self.model(**inputs)
                    # model_time += (time.time() - t0)
                    # t0 = time.time()
                    mfp_loss = outputs[0]
                    loss = mfp_loss
                    loss.backward()
                    # back_time += (time.time() - t0)
                    # t0 = time.time()
                    step_loss = loss.item()
                    step_acc = outputs[2] / outputs[1]
                    if self.args.local_rank in [-1, 0]:
                        pbar.set_description(f'epoch-{epoch}, mfp_loss={mfp_loss:.4f}, step_acc={step_acc:.4f}')

                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    # metric_time += (time.time() - t0)
                    # t0 = time.time()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                    tr_loss += step_loss
                    tr_acc += step_acc
                    # optim_time += (time.time() - t0)
                    # t0 = time.time()
                    if self.global_step % self.args.logging_steps == 0:
                        _log = {'window_loss': (tr_loss - logging_loss) / self.args.logging_steps,
                                'window_acc': (tr_acc - logging_acc) / self.args.logging_steps,
                                'time_cost': time.time() - start_time,}
                        if is_wandb_available() and self.args.local_rank:
                            wandb.log(_log, step=self.global_step)
                        logger.info(f'step = {self.global_step}, mfp_loss = {_log["window_loss"]}, mfp_acc = {_log["window_acc"]}, time_cost = {_log["time_cost"]}')
                        logging_loss = tr_loss
                        logging_acc = tr_acc
                        start_time = time.time()
                        # print('mask_time', mask_time / self.args.logging_steps)
                        # print('  pre_index_time', self.pre_index_time / self.args.logging_steps)
                        # print('  index_time', self.index_time / self.args.logging_steps)
                        # print('  post_index_time', self.post_index_time / self.args.logging_steps)
                        # print('todevice_time', todevice_time / self.args.logging_steps)
                        # print('model_time', model_time / self.args.logging_steps)
                        # print('  trans_time', self.model.trans_time / self.args.logging_steps)
                        # print('  proj_time', self.model.proj_time / self.args.logging_steps)
                        # print('  repeat_time', self.model.repeat_time / self.args.logging_steps)
                        # print('  criterion_time', self.model.criterion_time / self.args.logging_steps)
                        # print('  acc_compute_time', self.model.acc_compute_time / self.args.logging_steps)
                        # print('back_time', back_time / self.args.logging_steps)
                        # print('metric_time', metric_time / self.args.logging_steps)
                        # print('optim_time', optim_time / self.args.logging_steps)
                        # assert 0
                        # break
                
                if self.args.local_rank in [-1, 0]:
                    self.MFP_pretrain_eval()
                # break
            if self.args.local_rank in [-1, 0]:
                self.save_model(self.args.output_dir)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=['mfp_loss', 'mfp_acc'])))
        
    def MFP_pretrain_eval(self):
        eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        logger.info('***** running eval *****')
        logger.info(f'  num examples = {len(eval_dataloader.dataset)}')
        logger.info(f'  batch size = {self.args.eval_batch_size}')
        total_mfp_loss, total_mfp_acc, count = 0., 0., 0
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            for inputs in eval_dataloader:
                inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)

                mfp_loss, sample_num, acc = outputs[:3]
                count += sample_num
                total_mfp_loss += mfp_loss.item() * sample_num
                total_mfp_acc += acc

        eval_time_cost = time.time() - start_time
        eval_mfp_loss = total_mfp_loss / count
        eval_mfp_acc = total_mfp_acc / count
        self.eval_metrics.append([eval_mfp_loss, eval_mfp_acc])
        _log = {'learning_rate': self.scheduler.get_last_lr()[0], 
                'eval_mfp_loss': eval_mfp_loss, 'eval_mfp_acc': eval_mfp_acc, 'eval_time_cost': eval_time_cost}
        logger.info(str(_log))
        if is_wandb_available() and self.args.local_rank:
            wandb.log(_log, step=self.global_step)
        return _log

    def RFD_pretrain(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        # TODO add cosine scheduler
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info('***** running pretraining *****')
        logger.info(f'  dataset_name = {self.args.dataset_name}')
        logger.info(f'  input_size = {self.model_config.input_size}')
        logger.info(f'  num_fields = {self.model_config.num_fields}')
        logger.info(f'  num_examples = {len(self.train_dataset)}')
        logger.info(f'  num_epochs = {self.args.num_train_epochs}')
        logger.info(f'  batch_size = {self.args.train_batch_size}')
        logger.info(f'  n_gpu = {self.args.n_gpu}')
        logger.info(f'  per_gpu_train_batch_size = {self.args.per_gpu_train_batch_size}')
        logger.info(f'  total_steps = {t_total}')
        logger.info(f'  warmup_steps = {t_warmup}')
        logger.info(f'  learning_rate = {self.args.learning_rate}')
        logger.info(f'  weight_decay = {self.args.weight_decay}')
        logger.info(f'  lr_sched = {self.args.lr_sched}')
        logger.info(f'  mask_ratio = {self.args.mask_ratio}')
        logger.info(f'  pt_loss = {self.model_config.pt_loss}')
        logger.info(f'  pt_neg_num = {self.model_config.pt_neg_num}')
        logger.info(f'  share_neg = {self.model_config.share_neg}')
        logger.info(f'  pt_type = {self.model_config.pt_type}')
        logger.info(f'  gumbel_temp = {self.model_config.gumbel_temp}')
        logger.info(f'  G_w = {self.model_config.G_w}')
        logger.info(f'  D_w = {self.model_config.D_w}')
        logger.info(f'  RFD_G = {self.args.RFD_G}')
        if self.args.RFD_G == 'Model':
            self.model.generator.validate_model_config()
        self.model.discriminator.validate_model_config()
        tr_mfp_loss, logging_mfp_loss = 0., 0.
        tr_rfd_loss, logging_rfd_loss = 0., 0.
        tr_mfp_acc, logging_mfp_acc = 0., 0.
        tr_rfd_acc, logging_rfd_acc = 0., 0.

        self.global_step = 0
        self.eval_metrics = []

        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model.zero_grad()

        start_time = time.time()
        with trange(self.args.num_train_epochs, desc='epoch', ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f'-------------------- epoch-{epoch} --------------------')
                self.model.train()
                for step, inputs in enumerate(self.train_dataloader):
                    inputs = self.dynamic_mask(inputs, self.args.sampling_method)

                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)

                    outputs = self.model(**inputs)
                    loss, G_loss, G_acc, G_count, D_loss, D_acc, D_count, D_input_pos_ratio = outputs
                    loss.backward()
                    
                    step_mfp_loss, step_rfd_loss = G_loss.item(), D_loss.item()
                    step_mfp_acc, step_rfd_acc = G_acc.item(), D_acc.item()
                    if self.args.local_rank in [-1, 0]:
                        if self.args.RFD_G == 'Unigram':
                            pbar.set_description(f'D_loss={step_rfd_loss:.4f}, D_acc={step_rfd_acc:.4f}, pos_ratio={D_input_pos_ratio:.4f}')
                        else:
                            pbar.set_description(f'L={loss:.4f}, GL={step_mfp_loss:.4f}, Gacc={step_mfp_acc:.4f}, DL={step_rfd_loss:.4f}, Dacc={step_rfd_acc:.4f}, pos={D_input_pos_ratio:.4f}')

                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                    tr_mfp_loss += step_mfp_loss
                    tr_rfd_loss += step_rfd_loss
                    tr_mfp_acc += step_mfp_acc
                    tr_rfd_acc += step_rfd_acc
                    if self.global_step % self.args.logging_steps == 0:
                        _log = {'window_mfp_loss': (tr_mfp_loss - logging_mfp_loss) / self.args.logging_steps,
                                'window_rfd_loss': (tr_rfd_loss - logging_rfd_loss) / self.args.logging_steps,
                                'window_mfp_acc': (tr_mfp_acc - logging_mfp_acc) / self.args.logging_steps,
                                'window_rfd_acc': (tr_rfd_acc - logging_rfd_acc) / self.args.logging_steps,
                                'time_cost': time.time() - start_time,}
                        if is_wandb_available() and self.args.local_rank:
                            wandb.log(_log, step=self.global_step)
                        logger.info(f'step = {self.global_step}, mfp_loss = {_log["window_mfp_loss"]}, mfp_acc = {_log["window_mfp_acc"]}, rfd_loss = {_log["window_rfd_loss"]}, rfd_acc = {_log["window_rfd_acc"]}, time_cost = {_log["time_cost"]}')
                        logging_mfp_loss = tr_mfp_loss
                        logging_rfd_loss = tr_rfd_loss
                        logging_mfp_acc = tr_mfp_acc
                        logging_rfd_acc = tr_rfd_acc
                        start_time = time.time()
                        
                if self.args.local_rank in [-1, 0]:
                    self.RFD_pretrain_eval()
                # break
            if self.args.local_rank in [-1, 0]:
                self.save_model(self.args.output_dir)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=['mfp_loss', 'mfp_acc', 'rfd_loss', 'rfd_acc'])))
        
    def RFD_pretrain_eval(self):
        eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        logger.info('***** running eval *****')
        logger.info(f'  num examples = {len(eval_dataloader.dataset)}')
        logger.info(f'  batch size = {self.args.eval_batch_size}')
        total_mfp_loss, total_rfd_loss, mfp_count = 0., 0., 0
        total_mfp_acc, total_rfd_acc, rfd_count = 0., 0., 0
        total_D_input_pos_ratio = 0.
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            for inputs in eval_dataloader:
                inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
            
                loss, G_loss, G_acc, G_count, D_loss, D_acc, D_count, D_input_pos_ratio = outputs
                total_mfp_loss += G_loss * G_count
                total_mfp_acc += G_acc * G_count
                mfp_count += G_count
                total_rfd_loss += D_loss * D_count
                total_rfd_acc += D_acc * D_count
                rfd_count += D_count
        
        eval_time_cost = time.time() - start_time
        eval_mfp_loss, eval_mfp_acc = total_mfp_loss.item() / mfp_count, total_mfp_acc.item() / mfp_count
        eval_rfd_loss, eval_rfd_acc = total_rfd_loss.item() / rfd_count, total_rfd_acc.item() / rfd_count
        eval_pos_ratio = total_D_input_pos_ratio / rfd_count
        self.eval_metrics.append([eval_mfp_loss, eval_mfp_acc, eval_rfd_loss, eval_rfd_acc])
        _log = {'learning_rate': self.scheduler.get_last_lr()[0], 
                'eval_mfp_loss': eval_mfp_loss, 
                'eval_mfp_acc': eval_mfp_acc, 
                'eval_rfd_loss': eval_rfd_loss, 
                'eval_rfd_acc': eval_rfd_acc, 
                'eval_rfd_acc': eval_rfd_acc, 
                'eval_pos_ratio': eval_pos_ratio, 
                'eval_time_cost': eval_time_cost}
        logger.info(str(_log))
        if is_wandb_available() and self.args.local_rank:
            wandb.log(_log, step=self.global_step)
        return _log

    def SCARF_pretrain(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        # TODO add cosine scheduler
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info('***** running pretraining *****')
        logger.info(f'  dataset_name = {self.args.dataset_name}')
        logger.info(f'  input_size = {self.model_config.input_size}')
        logger.info(f'  num_fields = {self.model_config.num_fields}')
        logger.info(f'  num_examples = {len(self.train_dataset)}')
        logger.info(f'  num_epochs = {self.args.num_train_epochs}')
        logger.info(f'  batch_size = {self.args.train_batch_size}')
        logger.info(f'  n_gpu = {self.args.n_gpu}')
        logger.info(f'  per_gpu_train_batch_size = {self.args.per_gpu_train_batch_size}')
        logger.info(f'  total_steps = {t_total}')
        logger.info(f'  warmup_steps = {t_warmup}')
        logger.info(f'  learning_rate = {self.args.learning_rate}')
        logger.info(f'  weight_decay = {self.args.weight_decay}')
        logger.info(f'  lr_sched = {self.args.lr_sched}')
        logger.info(f'  mask_ratio = {self.args.mask_ratio}')
        logger.info(f'  pt_type = {self.model_config.pt_type}')
        self.model.validate_model_config()

        self.global_step = 0
        self.eval_metrics = []
        tr_loss, logging_loss = 0., 0.

        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model.zero_grad()

        start_time = time.time()
        with trange(self.args.num_train_epochs, desc='epoch', ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f'-------------------- epoch-{epoch} --------------------')
                self.model.train()
                for step, inputs in enumerate(self.train_dataloader):
                    inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    loss.backward()
                    step_loss = loss.item()
                    if self.args.local_rank in [-1, 0]:
                        pbar.set_description(f'epoch-{epoch}, scarf_loss={loss:.4f}')

                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                    tr_loss += step_loss
                    if self.global_step % self.args.logging_steps == 0:
                        _log = {'window_loss': (tr_loss - logging_loss) / self.args.logging_steps,
                                'time_cost': time.time() - start_time,}
                        if is_wandb_available() and self.args.local_rank:
                            wandb.log(_log, step=self.global_step)
                        logger.info(f'step = {self.global_step}, scarf_loss = {_log["window_loss"]}, time_cost = {_log["time_cost"]}')
                        logging_loss = tr_loss
                        start_time = time.time()
                        # break
                
                if self.args.local_rank in [-1, 0]:
                    self.SCARF_pretrain_eval()
                # break
            if self.args.local_rank in [-1, 0]:
                self.save_model(self.args.output_dir)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=['scarf_loss'])))
        
    def SCARF_pretrain_eval(self):
        eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        logger.info('***** running eval *****')
        logger.info(f'  num examples = {len(eval_dataloader.dataset)}')
        logger.info(f'  batch size = {self.args.eval_batch_size}')
        total_scarf_loss, count = 0., 0
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            for inputs in eval_dataloader:
                inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)

                scarf_loss, sample_num = outputs[:2]
                count += sample_num
                total_scarf_loss += scarf_loss.item() * sample_num

        eval_time_cost = time.time() - start_time
        eval_scarf_loss = total_scarf_loss / count
        self.eval_metrics.append([eval_scarf_loss])
        _log = {'learning_rate': self.scheduler.get_last_lr()[0], 
                'eval_scarf_loss': eval_scarf_loss, 'eval_time_cost': eval_time_cost}
        logger.info(str(_log))
        if is_wandb_available() and self.args.local_rank:
            wandb.log(_log, step=self.global_step)
        return _log

    def MF4UIP_pretrain(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        # TODO add cosine scheduler
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info('***** running pretraining *****')
        logger.info(f'  dataset_name = {self.args.dataset_name}')
        logger.info(f'  input_size = {self.model_config.input_size}')
        logger.info(f'  num_fields = {self.model_config.num_fields}')
        logger.info(f'  num_examples = {len(self.train_dataset)}')
        logger.info(f'  num_epochs = {self.args.num_train_epochs}')
        logger.info(f'  batch_size = {self.args.train_batch_size}')
        logger.info(f'  n_gpu = {self.args.n_gpu}')
        logger.info(f'  per_gpu_train_batch_size = {self.args.per_gpu_train_batch_size}')
        logger.info(f'  total_steps = {t_total}')
        logger.info(f'  warmup_steps = {t_warmup}')
        logger.info(f'  learning_rate = {self.args.learning_rate}')
        logger.info(f'  weight_decay = {self.args.weight_decay}')
        logger.info(f'  lr_sched = {self.args.lr_sched}')
        logger.info(f'  mask_ratio = {self.args.mask_ratio}')
        logger.info(f'  pt_type = {self.model_config.pt_type}')
        self.model.validate_model_config()

        self.global_step = 0
        self.eval_metrics = []
        tr_loss, logging_loss = 0., 0.

        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model.zero_grad()

        start_time = time.time()
        with trange(self.args.num_train_epochs, desc='epoch', ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f'-------------------- epoch-{epoch} --------------------')
                self.model.train()
                for step, inputs in enumerate(self.train_dataloader):
                    inputs = self.dynamic_mask(inputs, self.args.sampling_method)

                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    loss.backward()
                    step_loss = loss.item()
                    if self.args.local_rank in [-1, 0]:
                        pbar.set_description(f'epoch-{epoch}, mf4uip_loss={loss:.4f}')

                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                    tr_loss += step_loss
                    if self.global_step % self.args.logging_steps == 0:
                        _log = {'window_loss': (tr_loss - logging_loss) / self.args.logging_steps,
                                'time_cost': time.time() - start_time,}
                        if is_wandb_available() and self.args.local_rank:
                            wandb.log(_log, step=self.global_step)
                        logger.info(f'step = {self.global_step}, mf4uip_loss = {_log["window_loss"]}, time_cost = {_log["time_cost"]}')
                        logging_loss = tr_loss
                        start_time = time.time()
                        # break
                
                if self.args.local_rank in [-1, 0]:
                    self.MF4UIP_pretrain_eval()
                # break
            if self.args.local_rank in [-1, 0]:
                self.save_model(self.args.output_dir)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=['mf4uip_loss'])))
        
    def MF4UIP_pretrain_eval(self):
        eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        logger.info('***** running eval *****')
        logger.info(f'  num examples = {len(eval_dataloader.dataset)}')
        logger.info(f'  batch size = {self.args.eval_batch_size}')
        total_mf4uip_loss, count = 0., 0
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            for inputs in eval_dataloader:
                inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)

                mf4uip_loss, sample_num = outputs[:2]
                count += sample_num
                total_mf4uip_loss += mf4uip_loss.item() * sample_num

        eval_time_cost = time.time() - start_time
        eval_mf4uip_loss = total_mf4uip_loss / count
        self.eval_metrics.append([eval_mf4uip_loss])
        _log = {'learning_rate': self.scheduler.get_last_lr()[0], 
                'eval_mf4uip_loss': eval_mf4uip_loss, 'eval_time_cost': eval_time_cost}
        logger.info(str(_log))
        if is_wandb_available() and self.args.local_rank:
            wandb.log(_log, step=self.global_step)
        return _log

    def save_model(self, model_dir):
        if self.model_config.pt_type == 'MFP':
            save_dict = self.model.state_dict()
        elif self.model_config.pt_type == 'RFD':
            save_dict = self.model.discriminator.state_dict()
        elif self.model_config.pt_type == 'SCARF':
            save_dict = self.model.state_dict()
        elif self.model_config.pt_type == 'MF4UIP':
            save_dict = self.model.state_dict()
        torch.save(save_dict, os.path.join(model_dir, '{}.model'.format(self.global_step)))

    def load_model(self, load_step, model_dir):
        model_path = os.path.join(model_dir, '{}.model'.format(load_step))
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
    
    def test(self, test_dataset: Dataset, load_step=-1, model_dir=None):
        if load_step == -1:
            load_step = self.best_eval_step
        if model_dir is None:
            model_dir = self.args.output_dir
        self.load_model(load_step, model_dir)
        self.eval(test_dataset, test_eval=True)
