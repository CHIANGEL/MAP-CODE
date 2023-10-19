import copy
import json
import logging
import os
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import torch

from transformers.utils import cached_property

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = field(metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    dataset_name: str = field(default='avazu')
    data_dir: str = field(default='data/avazu', metadata={"help": "The root of data files."})
    per_gpu_train_batch_size: int = field(default=128, metadata={"help": "Batch size per GPU/CPU for training."})
    per_gpu_eval_batch_size: int = field(default=10000, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    adam_betas: str = field(default='0.9,0.999', metadata={'help': 'beta1 and beta2 for Adam optimizer.'})
    max_grad_norm: float = field(default=0.0, metadata={"help": "Max gradient norm. 0 for not clipping"})
    patience: int = field(default=2, metadata={"help": "The patience for early stop"})
    num_train_epochs: int = field(default=20, metadata={"help": "Total number of training epochs to perform."})
    lr_sched: str = field(default='cosine', metadata={"help": "Type of LR schedule method"})
    warmup_ratio: float = field(default=0.0,  metadata={"help": "Linear warmup over warmup_ratio if warmup_steps not set"})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=1000, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(default=20, metadata={"help": (
        "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. "
        "Default is unlimited checkpoints")})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    # Arguments for pretraining
    sampling_method: str = field(default='normal', metadata={"help": "The sampling method for the masked index"}, )
    mask_ratio: float = field(default=0.1, metadata={"help": "The proportion of field to be masked"})
    pretrain: bool = field(default=False, metadata={"help": "Whether to pretrain the model."}, )
    pt_type: str = field(default='MFP', metadata={"help": "The type of pretraining method: MFP, RFD"}, )
    RFD_replace: str = field(default='Unigram', metadata={"help": "The type of generator for RFD"}, )
    finetune: bool = field(default=False, metadata={"help": "Whether to finetune the model."}, )
    pretrained_model_path: str = field(default=None, metadata={"help": "The path of model to be finetuned"}, )

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


@dataclass
class ModelArguments:
    model_name: str
    # initializer_range: float = field(default=0.02)

    embed_size: int = field(default=32)
    embed_dropout_rate: float = field(default=0.0)
    hidden_size: int = field(default=128)
    num_hidden_layers: int = field(default=1)
    hidden_act: str = field(default='relu')
    hidden_dropout_rate: float = field(default=0.0)

    num_attn_heads: int = field(default=1)
    attn_probs_dropout_rate: float = field(default=0.1)
    intermediate_size: int = field(default=128)
    norm_first: bool = field(default=False)
    layer_norm_eps: float = field(default=1e-12)

    agg_type: str = field(default='mean')
    res_conn: bool = field(default=False)
    num_channels: int = field(default=1)
    embed_norm: bool = field(default=False)
    prod_layer_norm: bool = field(default=False)
    prod_dropout_rate: float = field(default=0.1)
    inter_layer_norm: bool = field(default=False)
    output_reduction: str = field(default='sum,max,sum')

    num_cross_layers: int = field(default=1)
    share_embedding: bool = field(default=False)
    channels: str = field(default='14,16,18,20')
    kernel_heights: str = field(default='7,7,7,7')
    pooling_sizes: str = field(default='2,2,2,2')
    recombined_channels: str = field(default='3,3,3,3')
    conv_act: str = field(default='tanh')
    reduction_ratio: int = field(default=3)
    bilinear_type: str = field(default='field_interaction')
    reuse_graph_layer: bool = field(default=False)
    attn_scale: bool = field(default=False)
    use_lr: bool = field(default=False)
    attn_size: int = field(default=40)
    num_attn_layers: int = field(default=2)
    cin_layer_units: str = field(default='50,50')
    field_interaction_type: str = field(default='matrixed')
    product_type: str = field(default='inner')
    outer_product_kernel_type: str = field(default='mat')

    # Arguments for loss in pretraining
    pt_neg_num: int = field(default=25, metadata={"help": "The number of negative features to be sampled for each instance"})
    proj_size: int = field(default=32, metadata={"help": "The project size for baseline models in pretraining phase"})

    # Additional arguments for transformer dnn
    dnn_size: int = field(default=1000, metadata={'help': "The size of each dnn layer"})
    num_dnn_layers: int = field(default=0, metadata={"help": "The number of dnn layers"})
    dnn_act: str = field(default='relu', metadata={'help': "The activation function for dnn layers"})
    dnn_drop: float = field(default=0.0, metadata={'help': "The dropout for dnn layers"})

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, save_directory):
        assert os.path.isdir(save_directory), f"not a directory: {save_directory}"
        output_config_file = os.path.join(save_directory, 'config.json')
        self.to_json_file(output_config_file)

    @classmethod
    def load(cls, load_directory):
        output_config_file = os.path.join(load_directory, 'config.json')
        config_dict = cls.from_json_file(output_config_file)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        config = cls(**config_dict)
        return config

    @classmethod
    def from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
