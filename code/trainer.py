import logging
import os, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from typing import List, Dict
import pandas as pd
from tqdm import tqdm, trange
from sklearn.metrics import log_loss, roc_auc_score
import time
import gc
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from arguments import Config, TrainingArguments
from models import BaseModel

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, 
        model: BaseModel, 
        model_config: Config, 
        training_args: TrainingArguments, 
        train_dataset: Dataset, 
        eval_dataset: Dataset, 
    ):
        self.model = model
        self.model_config = model_config
        self.args = training_args
        self.device = self.args.device
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.global_step = 0
        self.eval_metrics = []
        logger.info(f"setting device {self.device}")
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.best_eval_auc = 0
        self.best_eval_step = -1

    def get_dataloader(self, dataset, is_training=True):
        batch_size = self.args.per_gpu_train_batch_size if is_training else self.args.per_gpu_eval_batch_size
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=is_training, 
        )
        return dataloader

    def get_optimizer(self, num_training_steps: int, num_warmup_steps: int):
        no_decay = ["bias", "LayerNorm.weight"]
        named_params = [(k, v) for k, v in self.model.named_parameters()]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        beta1, beta2 = self.args.adam_betas.split(",")
        beta1, beta2 = float(beta1), float(beta2)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon,
                          betas=(beta1, beta2))
        if self.args.lr_sched.lower() == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)
        elif self.args.lr_sched.lower() == "const":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
        else:
            raise NotImplementedError

        return optimizer, scheduler

    def train(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info("\n***** running training *****")
        logger.info(f"  dataset_name = {self.args.dataset_name}")
        logger.info(f"  input_size = {self.model_config.input_size}")
        logger.info(f"  num_fields = {self.model_config.num_fields}")
        logger.info(f"  num_examples = {len(self.train_dataset)}")
        logger.info(f"  num_epochs = {self.args.num_train_epochs}")
        logger.info(f"  batch_size = {self.args.train_batch_size}")
        logger.info(f"  total_steps = {t_total}")
        logger.info(f"  warmup_steps = {t_warmup}")
        logger.info(f"  learning_rate = {self.args.learning_rate}")
        logger.info(f"  weight_decay = {self.args.weight_decay}")
        logger.info(f"  lr_sched = {self.args.lr_sched}")
        self.model.validate_model_config()

        self._patience = 0
        self._stop_training = False
        self.global_step = 0
        self.eval_metrics = []
        tr_loss, logging_loss = 0., 0.
        tr_labels, tr_probs = [], []

        self.model.to(self.device)
        self.model.zero_grad()

        with trange(self.args.num_train_epochs, desc="epoch", ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f"-------------------- epoch-{epoch} --------------------")
                self.model.train()
                for step, (X, Y) in enumerate(self.train_dataloader):
                    inputs = {
                        "input_ids": X,
                        "labels": Y
                    }
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    loss.backward()
                    step_loss = loss.item()
                    tr_probs.extend(torch.sigmoid(outputs[1].detach().cpu()).numpy())
                    tr_labels.extend(inputs["labels"].detach().cpu().numpy())
                    pbar.set_description(f"epoch-{epoch}, loss={step_loss:.4f}")

                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    tr_loss += step_loss
                    if self.global_step % self.args.logging_steps == 0:
                        window_auc = roc_auc_score(np.int32(tr_labels), np.array(tr_probs))
                        _log = {
                            "window_auc": window_auc,
                            "window_loss": (tr_loss - logging_loss) / self.args.logging_steps, 
                        }
                        tr_labels, tr_probs = [], []
                        logger.info(f"step = {self.global_step}, {str(_log)}")
                        logging_loss = tr_loss

                self.eval()
                if self._stop_training:
                    break

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=["auc", "log_loss"])))

    def eval(self, eval_dataset=None, test_eval=False):
        if eval_dataset is None:
            eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        else:
            eval_dataloader = self.get_dataloader(eval_dataset, is_training=False)
        if test_eval:
            logger.info("\n***** running TEST *****")
        else:
            logger.info("\n***** running eval *****")
        logger.info(f"  num examples = {len(eval_dataloader.dataset)}")
        logger.info(f"  batch size = {self.args.eval_batch_size}")
        eval_losses, eval_size, preds, probs, label_ids = 0, 0, [], [], []
        self.model.eval()

        for (X, Y) in eval_dataloader:
            with torch.no_grad():
                inputs = {
                    "input_ids": X,
                    "labels": Y
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                step_eval_loss, logits = outputs[:2]
                eval_losses += step_eval_loss.mean().item() * logits.size()[0]
                eval_size += logits.size()[0]
            preds.extend(logits.detach().cpu().numpy())
            probs.extend(torch.sigmoid(logits.detach().cpu()).numpy())
            label_ids.extend(inputs["labels"].detach().cpu().numpy())

        preds, probs, label_ids = np.array(preds).astype("float64"), np.array(probs).astype("float64"), np.array(label_ids)
        auc = roc_auc_score(y_true=label_ids, y_score=probs)
        ll = log_loss(y_true=label_ids, y_pred=probs)
        
        self.eval_metrics.append([auc, ll])
        _log = {
            "learning_rate": self.scheduler.get_last_lr()[0], 
            "eval_auc": auc, 
            "eval_loss": ll,
            "avg_logits": preds.mean(), 
            "avg_probs": probs.mean(), 
        }
        logger.info(str(_log))
        if test_eval == False:
            if auc > self.best_eval_auc:
                self.best_eval_auc = auc
                self.best_eval_step = self.global_step
                self._patience = 0
                self.save_model(self.args.output_dir)
            else:
                self._patience += 1
            if self._patience > self.args.patience:
                self._stop_training = True

    def dynamic_mask(self, inputs, sampling_method="normal"):
        batch_size = inputs["input_ids"].shape[0]
        num_fields = self.model_config.num_fields
        mask_num = int(num_fields * self.args.mask_ratio)
        if sampling_method == "normal":
            masked_index = torch.stack([torch.randperm(num_fields)[:mask_num] for i in range(batch_size)], dim=0)
        elif sampling_method == "randint":
            masked_index = torch.randint(0, num_fields, (batch_size, mask_num), device=inputs["input_ids"].device)
        else:
            raise NotImplementedError(sampling_method)

        if self.args.pt_type == "MFP":
            mask_ids = torch.zeros(masked_index.size(), dtype=torch.int64, device=inputs["input_ids"].device).fill_(3)
            inputs["labels"] = torch.gather(inputs["input_ids"], 1, masked_index)
            inputs["input_ids"] = torch.scatter(inputs["input_ids"], 1, masked_index, mask_ids)
            inputs["masked_index"] = masked_index
        elif self.args.pt_type == "RFD":
            if self.args.RFD_replace == "Unigram":
                sample_index = torch.randint(0, len(self.train_dataset), (batch_size * mask_num, ))
                replace_sample = torch.from_numpy(self.train_dataset.X[sample_index]).to(inputs["input_ids"].device)
                replace_feat = torch.gather(replace_sample, 1, masked_index.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs["input_ids"]
                inputs["input_ids"] = torch.scatter(inputs["input_ids"], 1, masked_index, replace_feat)
                inputs["labels"] = (origin_input_ids != inputs["input_ids"]).float()
            elif self.args.RFD_replace == "Uniform":
                replace_sample = torch.stack([torch.randint(self.model_config.idx_low[i], self.model_config.idx_high[i], (batch_size * mask_num, )) for i in range(num_fields)], dim=1)
                replace_feat = torch.gather(replace_sample, 1, masked_index.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs["input_ids"]
                inputs["input_ids"] = torch.scatter(inputs["input_ids"], 1, masked_index, replace_feat)
                inputs["labels"] = (origin_input_ids != inputs["input_ids"]).float()
            elif self.args.RFD_replace == "Whole-Uniform":
                replace_sample = torch.randint(10, self.model_config.input_size, (batch_size * mask_num, num_fields))
                replace_feat = torch.gather(replace_sample, 1, masked_index.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs["input_ids"]
                inputs["input_ids"] = torch.scatter(inputs["input_ids"], 1, masked_index, replace_feat)
                inputs["labels"] = (origin_input_ids != inputs["input_ids"]).float()
            elif self.args.RFD_replace == "Whole-Unigram":
                sample_index_0 = torch.randint(0, len(self.train_dataset), (batch_size * mask_num, ))
                replace_sample = torch.from_numpy(self.train_dataset.X[sample_index_0]).to(inputs["input_ids"].device)
                sample_index_1 = torch.randint(0, num_fields, (batch_size, mask_num), device=inputs["input_ids"].device)
                replace_feat = torch.gather(replace_sample, 1, sample_index_1.view(-1, 1)).view(batch_size, mask_num)
                origin_input_ids = inputs["input_ids"]
                inputs["input_ids"] = torch.scatter(inputs["input_ids"], 1, masked_index, replace_feat)
                inputs["labels"] = (origin_input_ids != inputs["input_ids"]).float()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(self.args.pt_type)

        return inputs

    def MFP_pretrain(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info("***** running pretraining *****")
        logger.info(f"  dataset_name = {self.args.dataset_name}")
        logger.info(f"  input_size = {self.model_config.input_size}")
        logger.info(f"  num_fields = {self.model_config.num_fields}")
        logger.info(f"  num_examples = {len(self.train_dataset)}")
        logger.info(f"  num_epochs = {self.args.num_train_epochs}")
        logger.info(f"  batch_size = {self.args.train_batch_size}")
        logger.info(f"  per_gpu_train_batch_size = {self.args.per_gpu_train_batch_size}")
        logger.info(f"  total_steps = {t_total}")
        logger.info(f"  warmup_steps = {t_warmup}")
        logger.info(f"  learning_rate = {self.args.learning_rate}")
        logger.info(f"  weight_decay = {self.args.weight_decay}")
        logger.info(f"  lr_sched = {self.args.lr_sched}")
        logger.info(f"  mask_ratio = {self.args.mask_ratio}")
        logger.info(f"  pt_neg_num = {self.model_config.pt_neg_num}")
        logger.info(f"  pt_type = {self.model_config.pt_type}")
        self.model.validate_model_config()

        self.global_step = 0
        self.eval_metrics = []
        tr_loss, logging_loss = 0., 0.
        tr_acc, logging_acc = 0., 0.

        self.model.to(self.device)
        self.model.zero_grad()

        start_time = time.time()
        with trange(self.args.num_train_epochs, desc="epoch", ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f"-------------------- epoch-{epoch} --------------------")
                self.model.train()
                for step, (X, Y) in enumerate(self.train_dataloader):
                    inputs = {
                        "input_ids": X,
                        "labels": Y
                    }
                    inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    outputs = self.model(**inputs)
                    
                    mfp_loss = outputs[0]
                    loss = mfp_loss
                    loss.backward()
                    
                    step_loss = loss.item()
                    step_acc = outputs[2] / outputs[1]
                    if self.args.local_rank in [-1, 0]:
                        pbar.set_description(f"epoch-{epoch}, mfp_loss={mfp_loss:.4f}, step_acc={step_acc:.4f}")

                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                    tr_loss += step_loss
                    tr_acc += step_acc
                    if self.global_step % self.args.logging_steps == 0:
                        _log = {
                            "window_loss": (tr_loss - logging_loss) / self.args.logging_steps,
                            "window_acc": (tr_acc - logging_acc) / self.args.logging_steps,
                            "time_cost": time.time() - start_time, 
                        }
                        logger.info(f"step = {self.global_step}, {str(log)})")
                        logging_loss = tr_loss
                        logging_acc = tr_acc
                        start_time = time.time()
                
                if self.args.local_rank in [-1, 0]:
                    self.MFP_pretrain_eval()
                
            if self.args.local_rank in [-1, 0]:
                self.save_model(self.args.output_dir)

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=["mfp_loss", "mfp_acc"])))
        
    def MFP_pretrain_eval(self):
        eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        logger.info("***** running eval *****")
        logger.info(f"  num examples = {len(eval_dataloader.dataset)}")
        logger.info(f"  batch size = {self.args.eval_batch_size}")
        total_mfp_loss, total_mfp_acc, count = 0., 0., 0
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            for X, Y in eval_dataloader:
                inputs = {
                    "input_ids": X,
                    "labels": Y
                }
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
        _log = {
            "learning_rate": self.scheduler.get_last_lr()[0], 
            "eval_mfp_loss": eval_mfp_loss, 
            "eval_mfp_acc": eval_mfp_acc, 
            "eval_time_cost": eval_time_cost, 
        }
        logger.info(str(_log))
        return _log

    def RFD_pretrain(self):
        self.train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(self.train_dataloader) * self.args.num_train_epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info("\n***** running pretraining *****")
        logger.info(f"  dataset_name = {self.args.dataset_name}")
        logger.info(f"  input_size = {self.model_config.input_size}")
        logger.info(f"  num_fields = {self.model_config.num_fields}")
        logger.info(f"  num_examples = {len(self.train_dataset)}")
        logger.info(f"  num_epochs = {self.args.num_train_epochs}")
        logger.info(f"  batch_size = {self.args.train_batch_size}")
        logger.info(f"  per_gpu_train_batch_size = {self.args.per_gpu_train_batch_size}")
        logger.info(f"  total_steps = {t_total}")
        logger.info(f"  warmup_steps = {t_warmup}")
        logger.info(f"  learning_rate = {self.args.learning_rate}")
        logger.info(f"  weight_decay = {self.args.weight_decay}")
        logger.info(f"  lr_sched = {self.args.lr_sched}")
        logger.info(f"  pt_type = {self.model_config.pt_type}")
        logger.info(f"  mask_ratio = {self.args.mask_ratio}")
        logger.info(f"  RFD_replace = {self.args.RFD_replace}")
        self.model.validate_model_config()
        tr_rfd_loss, logging_rfd_loss = 0., 0.
        tr_rfd_acc, logging_rfd_acc = 0., 0.

        self.global_step = 0
        self.eval_metrics = []

        self.model.to(self.device)
        self.model.zero_grad()

        start_time = time.time()
        with trange(self.args.num_train_epochs, desc="epoch", ncols=100) as pbar:
            for epoch in pbar:
                logger.info(f"-------------------- epoch-{epoch} --------------------")
                self.model.train()
                for step, (X, Y) in enumerate(self.train_dataloader):
                    inputs = {
                        "input_ids": X,
                        "labels": Y
                    }
                    inputs = self.dynamic_mask(inputs, self.args.sampling_method)

                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)

                    outputs = self.model(**inputs)
                    loss, count, acc, input_pos_ratio = outputs
                    loss.backward()
                    
                    step_rfd_loss, step_rfd_acc = loss.item(), acc.item()
                    if self.args.local_rank in [-1, 0]:
                        pbar.set_description(f"loss={step_rfd_loss:.4f}, acc={step_rfd_acc:.4f}, pos_ratio={input_pos_ratio:.4f}")

                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                    tr_rfd_loss += step_rfd_loss
                    tr_rfd_acc += step_rfd_acc
                    if self.global_step % self.args.logging_steps == 0:
                        _log = {
                            "window_rfd_loss": (tr_rfd_loss - logging_rfd_loss) / self.args.logging_steps,
                            "window_rfd_acc": (tr_rfd_acc - logging_rfd_acc) / self.args.logging_steps,
                            "time_cost": time.time() - start_time, 
                        }
                        logger.info(f"step = {self.global_step}, {str(_log)}")
                        logging_rfd_loss = tr_rfd_loss
                        logging_rfd_acc = tr_rfd_acc
                        start_time = time.time()
                        
                if self.args.local_rank in [-1, 0]:
                    self.RFD_pretrain_eval()
                
            if self.args.local_rank in [-1, 0]:
                self.save_model(self.args.output_dir)

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            logger.info(str(pd.DataFrame(self.eval_metrics, columns=["rfd_loss", "rfd_acc"])))
        
    def RFD_pretrain_eval(self):
        eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        logger.info("\n***** running eval *****")
        logger.info(f"  num examples = {len(eval_dataloader.dataset)}")
        logger.info(f"  batch size = {self.args.eval_batch_size}")
        total_rfd_loss, total_rfd_acc, rfd_count = 0., 0., 0
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            for X, Y in eval_dataloader:
                inputs = {
                    "input_ids": X,
                    "labels": Y
                }
                inputs = self.dynamic_mask(inputs, self.args.sampling_method)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
            
                loss, count, acc = outputs[:3]
                total_rfd_loss += loss * count
                total_rfd_acc += acc * count
                rfd_count += count
        
        eval_time_cost = time.time() - start_time
        eval_rfd_loss = total_rfd_loss.item() / rfd_count
        eval_rfd_acc = total_rfd_acc.item() / rfd_count
        self.eval_metrics.append([eval_rfd_loss, eval_rfd_acc])
        _log = {
            "learning_rate": self.scheduler.get_last_lr()[0], 
            "eval_rfd_loss": eval_rfd_loss, 
            "eval_rfd_acc": eval_rfd_acc, 
            "eval_time_cost": eval_time_cost, 
        }
        logger.info(str(_log))
        return _log
    
    def save_model(self, model_dir):
        save_dict = self.model.state_dict()
        torch.save(save_dict, os.path.join(model_dir, "{}.model".format(self.global_step)))

    def load_model(self, load_step, model_dir):
        model_path = os.path.join(model_dir, "{}.model".format(load_step))
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
