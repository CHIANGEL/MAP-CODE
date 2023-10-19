import os
import json
import logging
import torch
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from utils import cached_property

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    dataset_name: str = field(default='avazu_x4')
    split_name: str = field(default='x4-full')

    wandb_name: str = field(default='avazu_x4',
        metadata={"help": "The project name for the WANDB logging."})
    data_dir: str = field(default='../data/avazu/avazu_x4', metadata={"help": "The root of data files."})
    per_gpu_train_batch_size: int = field(default=128, metadata={"help": "Batch size per GPU/CPU for training."})
    per_gpu_eval_batch_size: int = field(default=10000, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-6, metadata={"help": "Epsilon for Adam optimizer."})
    adam_betas: str = field(default='0.9,0.98', metadata={'help': 'beta1 and beta2 for Adam optimizer.'})
    max_grad_norm: float = field(default=0.0, metadata={"help": "Max gradient norm. 0 for not clipping"})
    patience: int = field(default=2, metadata={"help": "The patience for early stop"})

    num_train_epochs: int = field(default=20, metadata={"help": "Total number of training epochs to perform."})
    # warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    lr_sched: str = field(default='cosine',
                                metadata={"help": "Type of LR schedule method"})
    warmup_ratio: float = field(default=0.0,
                                metadata={"help": "Linear warmup over warmup_ratio if warmup_steps not set"})

    # logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=1000, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(default=20, metadata={"help": (
        "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. "
        "Default is unlimited checkpoints")})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(default=False, metadata={
        "help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}, )
    fp16_opt_level: str = field(default="O1", metadata={"help": (
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
        "See details at https://nvidia.github.io/apex/amp.html")}, )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    # Arguments for pretraining
    sampling_method: str = field(default='normal', metadata={"help": "The sampling method for the masked index"}, )
    n_core: int = field(default=5, metadata={"help": "The n_core threshold for dataset"}, )
    mask_ratio: float = field(default=0.1, metadata={"help": "The proportion of field to be masked"})
    pretrain: bool = field(default=False, metadata={"help": "Whether to pretrain the model."}, )
    pt_type: str = field(default='MFP', metadata={"help": "The type of pretraining method: MFP, RFD"}, )
    RFD_G: str = field(default='Model', metadata={"help": "The type of generator for RFD"}, )
    finetune: bool = field(default=False, metadata={"help": "Whether to finetune the model."}, )
    finetune_model_path: str = field(default=None, metadata={"help": "The path of model to be finetuned"}, )

    @property
    def train_batch_size(self) -> int:
        return self.per_gpu_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        return self.per_gpu_eval_batch_size * max(1, self.n_gpu)

    @cached_property
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        n_gpu = torch.cuda.device_count()
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        # elif is_tpu_available():
        #     device = xm.xla_device()
        #     n_gpu = 0
        elif n_gpu == 1:
            # a specific subset of GPUs by `CUDA_VISIBLE_DEVICES=0`
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
        return device, n_gpu

    @property
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    def n_gpu(self):
        return self._setup_devices[1]

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
