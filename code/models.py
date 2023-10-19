import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import logging
from typing import Dict, Optional, Tuple
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import math
from nce import IndexLinear

from layers import Embeddings, InnerProductLayer, OuterProductLayer, MLPBlock, get_act, \
    ProductLayer, CrossNetV2, FGCNNBlock, SqueezeExtractionLayer, \
    BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, SelfAttention, \
    MultiChannelOutputHead, CIN, MultiHeadSelfAttention
from arguments import Config

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    used_params = []

    def __init__(self, model_name="BaseModel", config: Config=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.config = config

    @classmethod
    def from_config(cls, config: Config):
        model_name_lower = config.model_name.lower()
        if model_name_lower == "dnn":
            model_class = DNN
        elif model_name_lower == "autoint":
            model_class = AutoInt
        elif model_name_lower == "trans":
            model_class = Transformer
        elif model_name_lower == "fignn":
            model_class = FiGNN
        elif model_name_lower == "fgcnn":
            model_class = FGCNN
        elif model_name_lower == "deepfm":
            model_class = DeepFM
        elif model_name_lower == "xdeepfm":
            model_class = xDeepFM
        elif model_name_lower == "dcnv2":
            model_class = DCNV2
        else:
            raise NotImplementedError(config.model_name)
        model = model_class(config)
        return model

    def validate_model_config(self):
        if self.model_name.lower() in ["trans"]:
            assert self.config.embed_size == self.config.hidden_size, \
                f"model {self.model_name} requires embed_size == hidden_size"

        logger.info(f"  model_name = {self.model_name}")
        for key in self.used_params:
            logger.info(f"  {key} = {getattr(self.config, key)}")

    def get_outputs(self, inputs, labels=None, masked_index=None, is_pretrain=None):
        """
        Input:
            inputs: [batch size, input_dim] or [batch_size, ]

        Return:
            MFP: (loss, signal count, sum of accuracy)
            RFD: (loss, signal count, sum of accuracy, pos ratio)
        """
        batch_size = inputs.shape[0]
        if (is_pretrain is None and self.config.pretrain) or is_pretrain:
            if self.config.pt_type == "MFP": # Masked Feature Prediction
                enc_output = self.feat_encoder(inputs).view(batch_size, self.config.num_fields, self.config.proj_size)
                selected_output = torch.gather(enc_output, 1, masked_index.unsqueeze(-1).repeat(1, 1, self.config.proj_size))
                loss, logits, features = self.mfp_criterion(labels, selected_output)
                total_acc = (logits.argmax(dim=2) == 0).sum().item()
                outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
            elif self.config.pt_type == "RFD": # Replaced Feature Detection
                logits = self.pred_rfd(inputs)
                loss = self.rfd_criterion(logits, labels)
                count = labels.shape[0] * labels.shape[1]
                acc = ((torch.sigmoid(logits) > 0.5).float() == labels).sum() / count
                input_pos_ratio = labels.mean()
                outputs = (loss, count, acc, input_pos_ratio)
            else:
                raise NotImplementedError
        else: # Finetune or Train from Scratch
            outputs = (inputs,)
            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(inputs.view(-1), labels.float())
                outputs = (loss,) + outputs

        return outputs

    def load_from_target_model(self, target_model_dict):
        model_dict = self.state_dict()
        count = 0
        for k, v in target_model_dict.items():
            if k in model_dict and model_dict[k].shape == target_model_dict[k].shape:
                model_dict[k] = v
                print(f"Load tensor: {k}, {v.shape}")
            else:
                print(f"Unmatched tensor in the target model: {k}, {v.shape}")
                count += 1
        self.load_state_dict(model_dict)

    def load_for_finetune(self, model_path):
        pretrained_dict = torch.load(model_path)
        # print(len(pretrained_dict))
        self.load_from_target_model(pretrained_dict)

    def create_pretraining_predictor(self, input_dim):
        if self.config.pt_type == "MFP": # Masked Feature Prediction
            self.feat_encoder = nn.Linear(input_dim, self.config.num_fields * self.config.proj_size)
            self.mfp_criterion = IndexLinear(self.config)
        elif self.config.pt_type == "RFD": # Replaced Feature Detection
            self.pred_rfd = nn.Sequential(
                nn.Linear(input_dim, self.config.num_fields * self.config.proj_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.num_fields * self.config.proj_size, self.config.num_fields),
            )
            self.rfd_criterion = BCEWithLogitsLoss()
        else:
            raise NotImplementedError


class LR(BaseModel):
    used_params = []

    def __init__(self, config: Config):
        super().__init__(model_name="LR", config=config)
        self.embed_w = nn.Embedding(config.input_size, embedding_dim=1)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_ids, labels=None):
        batch_size = input_ids.shape[0]
        wx = self.embed_w(input_ids)
        logits = wx.sum(dim=1) + self.bias
        outputs = self.get_outputs(logits, labels, is_pretrain=False)
        
        return outputs


class FM(BaseModel):
    used_params = ["embed_size"]

    def __init__(self, config: Config):
        super().__init__(model_name="FM", config=config)
        self.lr_layer = LR(config)
        self.embed = Embeddings(config)
        self.ip_layer = InnerProductLayer(num_fields=config.num_fields)

    def forward(self, input_ids, labels=None, masked_index=None):
        feat_embed = self.embed(input_ids)
        lr_logits = self.lr_layer(input_ids)[0]
        dot_sum = self.ip_layer(feat_embed)
        logits = dot_sum + lr_logits
        outputs = self.get_outputs(logits, labels, is_pretrain=False)
        return outputs


class DNN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act"]

    def __init__(self, config: Config):
        super(DNN, self).__init__(model_name="DNN", config=config)

        self.embed = Embeddings(config)
        self.dnn = MLPBlock(input_dim=config.embed_size * config.num_fields,
                            hidden_size=config.hidden_size,
                            num_hidden_layers=config.num_hidden_layers,
                            hidden_dropout_rate=config.hidden_dropout_rate,
                            hidden_act=config.hidden_act)
        if config.pretrain:
            final_dim = config.hidden_size
            self.create_pretraining_predictor(final_dim)
        else:
            self.fc_out = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        nn_output = self.dnn(torch.flatten(feat_embed, 1))
        
        if self.config.pretrain:
            outputs = self.get_outputs(nn_output, labels, masked_index)
        else:
            logits = self.fc_out(nn_output)
            outputs = self.get_outputs(logits, labels)
            
        return outputs


class DeepFM(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act"]

    def __init__(self, config: Config):
        super(DeepFM, self).__init__(model_name="DeepFM", config=config)
        
        self.embed = Embeddings(config)
        self.lr_layer = LR(config)
        self.dnn = MLPBlock(input_dim=config.num_fields * config.embed_size,
                            hidden_size=config.hidden_size,
                            num_hidden_layers=config.num_hidden_layers,
                            hidden_dropout_rate=config.hidden_dropout_rate,
                            hidden_act=config.hidden_act)
        self.ip_layer = InnerProductLayer(num_fields=config.num_fields)
        if config.pretrain:
            final_dim = config.hidden_size + 1
            self.create_pretraining_predictor(final_dim)
        else:
            self.dnn_fc_out = nn.Linear(config.hidden_size, 1)


    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        dnn_vec = self.dnn(feat_embed.flatten(start_dim=1))
        
        if self.config.pretrain:
            lr_fm = self.lr_layer(input_ids)[0] + self.ip_layer(feat_embed)
            final_vec = torch.cat([dnn_vec, lr_fm], dim=1)
            outputs = self.get_outputs(final_vec, labels, masked_index)
        else:
            logits = self.dnn_fc_out(dnn_vec)
            logits += self.lr_layer(input_ids)[0]
            logits += self.ip_layer(feat_embed)
            outputs = self.get_outputs(logits, labels)

        return outputs


class xDeepFM(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "cin_layer_units", "use_lr"]

    def __init__(self, config: Config):
        super(xDeepFM, self).__init__(model_name="xDeepFM", config=config)
        
        self.embed = Embeddings(config)
        input_dim = config.num_fields * config.embed_size
        cin_layer_units = [int(c) for c in config.cin_layer_units.split(",")]
        self.cin = CIN(config.num_fields, cin_layer_units)
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(input_dim=input_dim,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                hidden_dropout_rate=config.hidden_dropout_rate,
                                hidden_act=config.hidden_act)
            final_dim = sum(cin_layer_units) + config.hidden_size
        else:
            self.dnn = None
            final_dim = sum(cin_layer_units)

        if config.pretrain:
            self.create_pretraining_predictor(final_dim)
        else:
            self.lr_layer = LR(config) if config.use_lr else None
            self.fc = nn.Linear(final_dim, 1)

    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        final_vec = self.cin(feat_embed)
        if self.dnn is not None:
            dnn_vec = self.dnn(feat_embed.flatten(start_dim=1))
            final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        
        if self.config.pretrain:
            outputs = self.get_outputs(final_vec, labels, masked_index)
        else:
            logits = self.fc(final_vec)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            outputs = self.get_outputs(logits, labels)

        return outputs


class DCNV2(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "num_cross_layers"]

    def __init__(self, config: Config):
        super(DCNV2, self).__init__(model_name="DCNV2", config=config)
        
        self.embed = Embeddings(config)
        input_dim = config.num_fields * config.embed_size
        self.cross_net = CrossNetV2(input_dim, config.num_cross_layers)
        if config.num_hidden_layers > 0:
            self.parallel_dnn = MLPBlock(input_dim=input_dim,
                                        hidden_size=config.hidden_size,
                                        num_hidden_layers=config.num_hidden_layers,
                                        hidden_dropout_rate=config.hidden_dropout_rate,
                                        hidden_act=config.hidden_act)
            final_dim = input_dim + config.hidden_size
        else:
            final_dim = input_dim
        if config.pretrain:
            self.create_pretraining_predictor(final_dim)
        else:
            self.fc_out = nn.Linear(final_dim, 1)

    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids).flatten(start_dim=1)
        cross_output = self.cross_net(feat_embed)
        if self.config.num_hidden_layers > 0:
            dnn_output = self.parallel_dnn(feat_embed)
            final_output = torch.cat([cross_output, dnn_output], dim=-1)
        else:
            final_output = cross_output
        
        if self.config.pretrain:
            outputs = self.get_outputs(final_output, labels, masked_index)
        else:
            logits = self.fc_out(final_output)
            outputs = self.get_outputs(logits, labels)

        return outputs


class FGCNN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "share_embedding", "channels", "kernel_heights", "pooling_sizes", "recombined_channels",
                   "conv_act"]

    def __init__(self, config: Config):
        super(FGCNN, self).__init__(model_name="fgcnn", config=config)
        
        self.share_embedding = config.share_embedding
        self.embed = Embeddings(config)
        if not self.share_embedding:
            self.fg_embed = Embeddings(config)
        channels = [int(c) for c in config.channels.split(",")]
        kernel_heights = [int(c) for c in config.kernel_heights.split(",")]
        pooling_sizes = [int(c) for c in config.pooling_sizes.split(",")]
        recombined_channels = [int(c) for c in config.recombined_channels.split(",")]
        self.fgcnn_layer = FGCNNBlock(config.num_fields,
                                      config.embed_size,
                                      channels=channels,
                                      kernel_heights=kernel_heights,
                                      pooling_sizes=pooling_sizes,
                                      recombined_channels=recombined_channels,
                                      activation=config.conv_act,
                                      batch_norm=True)
        final_dim, total_features = self.compute_input_dim(config.embed_size,
                                                           config.num_fields,
                                                           channels,
                                                           pooling_sizes,
                                                           recombined_channels)
        self.ip_layer = InnerProductLayer(total_features, output="inner_product")
        if config.pretrain:
            self.create_pretraining_predictor(final_dim)
        else:
            if config.num_hidden_layers > 0:
                self.dnn = MLPBlock(input_dim=final_dim,
                                    hidden_size=config.hidden_size,
                                    num_hidden_layers=config.num_hidden_layers,
                                    hidden_dropout_rate=config.hidden_dropout_rate,
                                    hidden_act=config.hidden_act)
                self.fc_out = nn.Linear(config.hidden_size, 1)
            else:
                self.dnn = None
                self.fc_out = nn.Linear(final_dim, 1)

    def compute_input_dim(self,
                          embedding_dim,
                          num_fields,
                          channels,
                          pooling_sizes,
                          recombined_channels):
        total_features = num_fields
        input_height = num_fields
        for i in range(len(channels)):
            input_height = int(math.ceil(input_height / pooling_sizes[i]))
            total_features += input_height * recombined_channels[i]
        final_dim = int(total_features * (total_features - 1) / 2) \
                  + total_features * embedding_dim
        return final_dim, total_features

    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        if not self.share_embedding:
            feat_embed2 = self.fg_embed(input_ids)
        else:
            feat_embed2 = feat_embed
        conv_in = torch.unsqueeze(feat_embed2, 1)
        new_feat_embed = self.fgcnn_layer(conv_in)
        combined_feat_embed = torch.cat([feat_embed, new_feat_embed], dim=1)
        ip_vec = self.ip_layer(combined_feat_embed)
        dense_input = torch.cat([combined_feat_embed.flatten(start_dim=1), ip_vec], dim=1)
        
        if self.config.pretrain:
            outputs = self.get_outputs(dense_input, labels, masked_index)
        else:
            if self.dnn is not None:
                nn_output = self.dnn(dense_input)
                logits = self.fc_out(nn_output)
            else:
                logits = self.fc_out(dense_input)
            outputs = self.get_outputs(logits, labels)

        return outputs


class FiGNN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "res_conn", "reuse_graph_layer"]

    def __init__(self, config: Config):
        super(FiGNN, self).__init__(model_name="FiGNN", config=config)
        logger.warning("this model requires embed_size == hidden_size, only uses embed_size")
        
        self.embed = Embeddings(config)
        self.fignn = FiGNNBlock(config)
        if config.pretrain:
            final_dim = config.num_fields * config.embed_size
            self.create_pretraining_predictor(final_dim)
        else:
            self.fc = AttentionalPrediction(config)

    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        h_out = self.fignn(feat_embed)
        
        if self.config.pretrain:
            outputs = self.get_outputs(h_out.flatten(start_dim=1), labels, masked_index)
        else:
            logits = self.fc(h_out)
            outputs = self.get_outputs(logits, labels)
        
        return outputs


class AutoInt(BaseModel):
    used_params = ["embed_size", "num_attn_layers", "attn_size", "num_attn_heads", 
                   "attn_probs_dropout_rate", "use_lr", "res_conn", "attn_scale", 
                   "dnn_size", "num_dnn_layers", "dnn_act", "dnn_drop"]

    def __init__(self, config: Config):
        super(AutoInt, self).__init__(model_name="AutoInt", config=config)
        
        self.embed = Embeddings(config)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(config.embed_size if i == 0 else config.num_attn_heads * config.attn_size,
                                     attention_dim=config.attn_size, 
                                     num_heads=config.num_attn_heads, 
                                     dropout_rate=config.attn_probs_dropout_rate, 
                                     use_residual=config.res_conn, 
                                     use_scale=config.attn_scale, 
                                     layer_norm=False,
                                     align_to="output") 
             for i in range(config.num_attn_layers)])
        final_dim = config.num_fields * config.attn_size * config.num_attn_heads

        if config.pretrain:
            self.create_pretraining_predictor(final_dim)
        else:
            self.attn_out = nn.Linear(final_dim, 1)
            self.lr_layer = LR(config) if config.use_lr else None
            self.dnn = MLPBlock(input_dim=final_dim,
                                hidden_size=config.dnn_size,
                                num_hidden_layers=config.num_dnn_layers,
                                hidden_dropout_rate=config.dnn_drop,
                                hidden_act=config.dnn_act) if config.num_dnn_layers else None
            self.dnn_out = nn.Linear(config.dnn_size, 1) if config.num_dnn_layers else None
            
    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        attention_out = self.self_attention(feat_embed)
        attention_out = torch.flatten(attention_out, start_dim=1)
        
        if self.config.pretrain:
            outputs = self.get_outputs(attention_out, labels, masked_index)
        else:
            logits = self.attn_out(attention_out)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            if self.dnn is not None:
                logits += self.dnn_out(self.dnn(feat_embed.flatten(start_dim=1)))
            outputs = self.get_outputs(logits, labels)
        return outputs


class Transformer(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "num_attn_heads", "intermediate_size", "output_reduction", 
                   "norm_first", "layer_norm_eps", "use_lr", 
                   "dnn_size", "num_dnn_layers", "dnn_act", "dnn_drop"]

    def __init__(self, config: Config):
        super(Transformer, self).__init__(model_name="trans", config=config)
        logger.warning("this model requires embed_size == hidden_size, only uses embed_size")
        
        self.embed = Embeddings(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attn_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_rate,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        if config.pretrain:
            final_dim = config.num_fields * config.embed_size
            self.create_pretraining_predictor(final_dim)
        else:
            if self.config.output_reduction == "fc":
                self.trans_out = nn.Linear(config.num_fields * config.embed_size, 1)
            elif self.config.output_reduction == "mean,fc" or self.config.output_reduction == "sum,fc":
                self.trans_out = nn.Linear(config.embed_size, 1)
            elif self.config.output_reduction == "attn,fc":
                self.field_reduction_attn = nn.Sequential(
                    nn.Linear(config.embed_size, config.embed_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.embed_size, 1),
                    nn.Softmax(dim=1),
                )
                self.trans_out = nn.Linear(config.embed_size, 1)
            else:
                raise NotImplementedError
            self.lr_layer = LR(config) if config.use_lr else None
            if config.num_dnn_layers > 0:
                self.mlp = MLPBlock(input_dim=config.num_fields * config.embed_size,
                                    hidden_size=config.dnn_size,
                                    num_hidden_layers=config.num_dnn_layers,
                                    hidden_dropout_rate=config.dnn_drop,
                                    hidden_act=config.dnn_act)
                self.mlp_out = nn.Linear(config.dnn_size, 1)
            else:
                self.mlp = None

    def forward(self, input_ids, labels=None, masked_index=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        enc_output = self.encoder(feat_embed)
        
        if self.config.pretrain:
            outputs = self.get_outputs(enc_output.flatten(start_dim=1), labels, masked_index)
        else:
            # Finetune or train from scratch
            if self.config.output_reduction == "fc":
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == "mean,fc":
                enc_output = torch.sum(enc_output, dim=1) / self.config.num_fields
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == "sum,fc":
                enc_output = torch.sum(enc_output, dim=1)
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == "attn,fc":
                attn_score = self.field_reduction_attn(enc_output)
                attn_feat = torch.sum(enc_output * attn_score, dim=1)
                logits = self.trans_out(attn_feat)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            if self.mlp is not None:
                logits += self.mlp_out(self.mlp(feat_embed.flatten(start_dim=1)))
            outputs = self.get_outputs(logits, labels)
        return outputs
