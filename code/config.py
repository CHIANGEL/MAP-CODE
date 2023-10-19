import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


logger = logging.getLogger(__name__)


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
    pun_embed: bool = field(default=False)
    res_conn: bool = field(default=False)
    num_channels: int = field(default=1)
    embed_norm: bool = field(default=False)
    prod_layer_norm: bool = field(default=False)
    prod_dropout_rate: float = field(default=0.1)
    inter_layer_norm: bool = field(default=False)
    output_reduction: str = field(default='sum,max,sum')
    output_dim: int = field(default=1)

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

    # Arguments for adversarial training
    adv_train: bool = field(default=False, metadata={"help": "Whether to train with adversarial samples"},)
    attack_method: str = field(default='fgsm', metadata={'help': 'Which adversairal attack method to be used'})
    adv_epsilon: float = field(default=0.05, metadata={"help": "The perturbation bound for adversrial attacks"})

    # Arguments for loss in pretraining
    pun_pt_channel_agg: str = field(default='sum', metadata={'help': "The type of aggregation of channel dimension for pretraining"})
    pt_loss: str = field(default='full', metadata={'help': "The type of loss for masked feature prediction"})
    pt_neg_num: int = field(default=25, metadata={"help": "The number of negative features to be sampled for each instance"})
    share_neg: bool = field(default=False, metadata={'help': "Whether to share the negative samples in a mini-batch"})
    proj_size: int = field(default=32, metadata={"help": "The project size for baseline models in pretraining phase"})
    gumbel_temp: float = field(default=1.0, metadata={"help": "The temperature for gumbel sampling in RFD."}, )
    G_w: float = field(default=1.0, metadata={"help": "The loss weight for G in RFD."}, )
    D_w: float = field(default=1.0, metadata={"help": "The temperature for D in RFD."}, )
    info_nce_temp: float = field(default=1.0, metadata={"help": "The temperature for InfoNCE loss."}, )

    # Additional arguments for transformer dnn
    dnn_size: int = field(default=1000, metadata={'help': "The size of each dnn layer"})
    num_dnn_layers: int = field(default=0, metadata={"help": "The number of dnn layers"})
    dnn_act: str = field(default='relu', metadata={'help': "The activation function for dnn layers"})
    dnn_drop: float = field(default=0.0, metadata={'help': "The dropout for dnn layers"})

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class Config:
    def __init__(
            self,
            model_name,
            # meta info
            data_dir,
            input_size,
            num_fields,
            pretrain,
            pt_type,
            feat_count,
            n_core,
            device,
            n_gpu,
            idx_low,
            idx_high,
            feat_num_per_field,
            is_generator=False,
            num_feat_types=1,
            # models with embedding layer
            embed_size=32,
            embed_dropout_rate=0.0,
            hidden_size=128,
            num_hidden_layers=1,
            hidden_act='relu',
            hidden_dropout_rate=0.0,
            # transformer models
            num_attn_heads=1,
            attn_probs_dropout_rate=0.1,
            intermediate_size=128,
            norm_first=False,
            layer_norm_eps=1e-12,
            # currently not used
            # initializer_range=0.02,
            # pun
            agg_type='mean',
            pun_embed=False,
            res_conn=False,
            num_channels=1,
            embed_norm=False,
            prod_layer_norm=False,
            prod_dropout_rate=0.1,
            inter_layer_norm=False,
            output_reduction='sum,sum,sum',
            output_dim=1,
            # dcn v2
            num_cross_layers=1,
            # fgcnn
            share_embedding=False,
            channels='14,16,18,20',
            kernel_heights=7,
            pooling_sizes=2,
            recombined_channels=2,
            conv_act='tanh',
            # conv_batch_norm=True,
            # fibinet
            reduction_ratio=3,
            bilinear_type='field_interaction',
            # fignn
            reuse_graph_layer=False,
            # autoint, interhat
            num_attn_layers=2,
            attn_scale = False,
            use_lr = False,
            attn_size = 40,
            # xdeepfm
            cin_layer_units='50,50',
            # FmFM
            field_interaction_type='matrixed',
            # PNN
            product_type='inner',
            outer_product_kernel_type='mat',
            # adversarial training
            adv_train=False,
            attack_method='fgm',
            adv_epsilon=0.05,
            # pretrain loss
            pun_pt_channel_agg='sum',
            pt_loss='full',
            pt_neg_num=25,
            share_neg=False,
            proj_size=32,
            gumbel_temp=1.0,
            G_w=1.0,
            D_w=1.0,
            RFD_G='Model',
            info_nce_temp=1.0,
            # Additional args for transformer dnn
            dnn_size=1000,
            num_dnn_layers=0,
            dnn_act='relu',
            dnn_drop=0.0,
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.input_size = input_size
        self.num_fields = num_fields
        self.pretrain = pretrain
        self.pt_type = pt_type
        self.is_generator = is_generator
        self.feat_count = feat_count
        self.n_core = n_core
        self.device = device
        self.n_gpu = n_gpu
        self.idx_low = idx_low
        self.idx_high = idx_high
        self.feat_num_per_field = feat_num_per_field
        self.num_feat_types = num_feat_types

        self.embed_size = embed_size
        self.embed_dropout_rate = embed_dropout_rate

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.hidden_dropout_rate = hidden_dropout_rate

        self.num_attn_heads = num_attn_heads
        self.attn_probs_dropout_rate = attn_probs_dropout_rate
        self.intermediate_size = intermediate_size
        self.norm_first = norm_first
        self.layer_norm_eps = layer_norm_eps

        # self.initializer_range = initializer_range
        self.agg_type = agg_type
        self.pun_embed = pun_embed
        self.res_conn = res_conn
        self.num_channels = num_channels
        self.embed_norm = embed_norm
        self.prod_layer_norm = prod_layer_norm
        self.prod_dropout_rate = prod_dropout_rate
        self.inter_layer_norm = inter_layer_norm
        self.output_reduction = output_reduction
        self.output_dim = output_dim

        self.num_cross_layers = num_cross_layers

        self.share_embedding = share_embedding
        self.channels = channels
        self.kernel_heights = kernel_heights
        self.pooling_sizes = pooling_sizes
        self.recombined_channels = recombined_channels
        self.conv_act = conv_act
        # self.conv_batch_norm = conv_batch_norm

        self.reduction_ratio = reduction_ratio
        self.bilinear_type = bilinear_type

        self.reuse_graph_layer = reuse_graph_layer

        self.num_attn_layers = num_attn_layers
        
        self.attn_scale = attn_scale
        self.attn_size = attn_size
        self.use_lr = use_lr
        
        self.cin_layer_units = cin_layer_units

        self.field_interaction_type = field_interaction_type

        self.product_type = product_type
        self.outer_product_kernel_type = outer_product_kernel_type

        self.adv_train = adv_train
        self.attack_method = attack_method
        self.adv_epsilon = adv_epsilon

        self.pun_pt_channel_agg = pun_pt_channel_agg
        self.pt_loss = pt_loss
        self.pt_neg_num = pt_neg_num
        self.proj_size = proj_size
        self.share_neg = share_neg
        self.gumbel_temp = gumbel_temp
        self.G_w = G_w
        self.D_w = D_w
        self.RFD_G = RFD_G
        self.info_nce_temp = info_nce_temp
        
        self.dnn_size = dnn_size
        self.num_dnn_layers = num_dnn_layers
        self.dnn_act = dnn_act
        self.dnn_drop = dnn_drop

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
