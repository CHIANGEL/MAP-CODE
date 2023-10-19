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
    PUNEmbeddings, ProductLayer, CrossNetV2, FGCNNBlock, SqueezeExtractionLayer, \
    BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, SelfAttention, \
    ProductLayerV5, MultiChannelOutputHead, ProductLayerV6, ProductLayerV4, \
    CIN, MultiHeadSelfAttention, InterHAt_AttentionalAggregation, InterHAt_MultiHeadAttention, \
    InterHAt_MultiHeadSelfAttention, InterHAt_FeedForwardNetwork
from config import Config

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    used_params = []

    def __init__(self, model_name='BaseModel', config: Config=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.config = config

    @classmethod
    def from_config(cls, config: Config):
        model_name_lower = config.model_name.lower()
        if model_name_lower == 'lr':
            model_class = LR
        elif model_name_lower == 'fm':
            model_class = FM
        elif model_name_lower == 'dnn':
            model_class = DNN
        elif model_name_lower == 'pun':
            model_class = PUN
        elif model_name_lower == 'trans':
            model_class = Transformer
        elif model_name_lower == 'fmfm':
            model_class = FmFM
        elif model_name_lower == 'ipnn':
            assert config.product_type.lower() == 'inner'
            model_class = PNN
        elif model_name_lower == 'opnn':
            assert config.product_type.lower() == 'outer'
            model_class = PNN
        elif model_name_lower == 'deepfm':
            model_class = DeepFM
        elif model_name_lower == 'xdeepfm':
            model_class = xDeepFM
        elif model_name_lower == 'dcnv2':
            model_class = DCNV2
        elif model_name_lower == 'fgcnn':
            model_class = FGCNN
        elif model_name_lower == 'fibinet':
            model_class = FiBiNet
        elif model_name_lower == 'fignn':
            model_class = FiGNN
        elif model_name_lower == 'autoint':
            model_class = AutoInt
        elif model_name_lower == 'interhat':
            model_class = InterHAt
        elif model_name_lower == 'punv5':
            model_class = PUNV5
        elif model_name_lower == 'punv6':
            model_class = PUNV6
        elif model_name_lower == 'punv4':
            model_class = PUNV4
        else:
            raise NotImplementedError
        model = model_class(config)
        return model

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and 'embedding' in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logger.info(f'total number of parameters: {total_params}')

    def validate_model_config(self):
        if self.model_name.lower() in ['lr', 'fm', 'dcnv2', ]:
            assert self.config.output_dim == 1, f'model {self.model_name} requires output_dim == 1'

        if self.model_name.lower() in ['pun', 'punv4', 'trans', 'punv5', 'punv6']:
            assert self.config.embed_size == self.config.hidden_size, \
                f'model {self.model_name} requires embed_size == hidden_size'

        logger.info(f'  model_name = {self.model_name}')
        for key in self.used_params:
            logger.info(f'  {key} = {getattr(self.config, key)}')

    def get_outputs(self, logits, labels=None):
        outputs = (logits,)

        if labels is not None:
            if self.config.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.config.output_dim)), labels.long())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels)
            outputs = (loss,) + outputs

        return outputs

    def load_from_target_model(self, target_model_dict):
        model_dict = self.state_dict()
        count = 0
        for k, v in target_model_dict.items():
            if k in model_dict and model_dict[k].shape == target_model_dict[k].shape:
                model_dict[k] = v
            else:
                print(f'Unmatched tensor in the target model: {k}, {v.shape}')
                count += 1
        self.load_state_dict(model_dict)

    def load_for_finetune(self, model_path):
        pretrained_dict = torch.load(model_path)
        # print(len(pretrained_dict))
        self.load_from_target_model(pretrained_dict)

    def create_pretraining_predictor(self, input_dim):
        if self.config.pt_type == 'MFP': # Masked Feature Prediction
            self.feat_encoder = nn.Linear(input_dim, self.config.num_fields * self.config.proj_size)
            self.criterion = IndexLinear(self.config)
        elif self.config.pt_type == 'RFD' and self.config.is_generator: # The generator for Replaced Feature Detection
            self.feat_encoder = nn.Linear(input_dim, self.config.num_fields * self.config.proj_size)
            self.criterion = IndexLinear(self.config)
        elif self.config.pt_type == 'RFD' and not self.config.is_generator: # The discriminator for Replaced Feature Detection
            self.pred_rfd = nn.Sequential(
                nn.Linear(input_dim, self.config.num_fields * self.config.proj_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.num_fields * self.config.proj_size, self.config.num_fields),
            )
            self.criterion = BCEWithLogitsLoss()
        elif self.config.pt_type == 'SCARF': # SCARF: Contrastive Learning
            self.feat_encoder = nn.Linear(input_dim, self.config.proj_size)
        elif self.config.pt_type == 'MF4UIP': # MF4UIP: Masked Field Prediction for User Intent Prediction
            self.feat_encoder = nn.Linear(input_dim, self.config.num_fields * self.config.proj_size)
            self.predictor = torch.nn.ModuleList([
                nn.Linear(self.config.proj_size, self.config.feat_num_per_field[i]) for i in range(self.config.num_fields)
            ])
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            raise NotImplementedError
    
    def get_pretraining_output(self, input_vec, labels, masked_index):
        '''
        Input:
            input_vec: [batch size, input_vec]

        Return:
            MFP: (loss, signal count, sum of accuracy)
            RFD: 
                Generator: (loss, signal count, logits, features)
                Discriminator: (loss, signal count, logits)
            SCARF: (loss, signal count)
            MF4UIP: (loss, signal count)
        '''
        batch_size = input_vec.shape[0]
        if self.config.pt_type == 'MFP': # Masked Feature Prediction
            enc_output = self.feat_encoder(input_vec).view(batch_size, self.config.num_fields, self.config.proj_size)
            selected_output = torch.gather(enc_output, 1, masked_index.unsqueeze(-1).repeat(1, 1, self.config.proj_size))
            loss, G_logits, G_features = self.criterion(labels, selected_output)
            total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
            outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        elif self.config.pt_type == 'RFD':
            if self.config.is_generator: # The generator for Replaced Feature Detection
                enc_output = self.feat_encoder(input_vec)
                selected_output = enc_output.unsqueeze(1).repeat(1, masked_index.shape[1], 1)
                loss, G_logits, G_features = self.criterion(labels, selected_output)
                outputs = (loss, labels.shape[0] * labels.shape[1], G_logits, G_features)
            else: # The discriminator for Replaced Feature Detection
                D_logits = self.pred_rfd(input_vec)
                loss = self.criterion(D_logits, labels)
                outputs = (loss, labels.shape[0] * labels.shape[1], D_logits)
        elif self.config.pt_type == 'SCARF':
            enc_output = self.feat_encoder(input_vec)
            outputs = self.info_nce_loss(enc_output)
        elif self.config.pt_type == 'MF4UIP':
            enc_output = self.feat_encoder(input_vec).view(batch_size, self.config.num_fields, self.config.proj_size)
            pred_logits = [self.predictor[i](enc_output[:, i, :]) for i in range(self.config.num_fields)]
            losses = torch.stack([self.criterion(pred_logits[i], labels[:, i]) for i in range(self.config.num_fields)], dim=1)
            loss = torch.gather(losses, 1, masked_index).mean()
            outputs = (loss, masked_index.shape[0] * masked_index.shape[1])
        return outputs

    def info_nce_loss(self, features):
        # InfoNCE for in-batch contrastive learning
        batch_size = features.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.config.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.config.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.config.device)

        logits = logits / self.config.info_nce_temp

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return loss, labels.shape[0]


class LR(BaseModel):
    used_params = ['output_dim']

    def __init__(self, config: Config):
        super().__init__(model_name='LR', config=config)
        self.embed_w = nn.Embedding(config.input_size, embedding_dim=1)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_ids, labels=None):
        batch_size = input_ids.shape[0]
        wx = self.embed_w(input_ids)
        logits = wx.sum(dim=1) + self.bias
        outputs = self.get_outputs(logits, labels)
        
        return outputs


class FM(BaseModel):
    used_params = ['embed_size', 'output_dim']

    def __init__(self, config: Config):
        super().__init__(model_name='FM', config=config)
        self.lr_layer = LR(config)
        self.embed = Embeddings(config)
        if config.pretrain:
            # self.ip_layer = InnerProductLayer(num_fields=config.num_fields, output='inner_product')
            # self.feat_encoder = nn.Linear(config.num_fields * (config.num_fields - 1) // 2 + config.num_fields + 1, config.proj_size)
            self.ip_layer = InnerProductLayer(num_fields=config.num_fields)
            self.feat_encoder = nn.Linear(1, config.proj_size)
            self.criterion = IndexLinear(config)
        else:
            self.ip_layer = InnerProductLayer(num_fields=config.num_fields)

    def forward(self, input_ids, labels=None, masked_index=None):
        feat_embed = self.embed(input_ids)
        
        if self.config.pretrain:
            # Pretrain phase
            lr_vec = self.lr_layer(input_ids)[0]
            fm_vec = self.ip_layer(feat_embed)
            # print(lr_vec.shape)
            # print(fm_vec.shape)
            enc_output = self.feat_encoder(torch.cat([lr_vec, fm_vec], dim=1))
            # print(enc_output.shape)
            # assert 0
            selected_output = enc_output.unsqueeze(1).repeat(1, masked_index.shape[1], 1)
            loss, G_logits, G_features = self.criterion(labels, selected_output)
            total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
            outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        else:
            lr_logits = self.lr_layer(input_ids)[0]
            dot_sum = self.ip_layer(feat_embed)
            logits = dot_sum + lr_logits
            outputs = self.get_outputs(logits, labels)

        return outputs


class DNN(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'output_dim', 'pretrain', 'pt_type', 'is_generator']

    def __init__(self, config: Config):
        super(DNN, self).__init__(model_name='DNN', config=config)
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
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
            self.fc_out = nn.Linear(config.hidden_size, config.output_dim)

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        nn_output = self.dnn(torch.flatten(feat_embed, 1))
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(nn_output, labels, masked_index)
        else:
            logits = self.fc_out(nn_output)
            outputs = self.get_outputs(logits, labels)

        return outputs


class FmFM(BaseModel):
    used_params = ['embed_size', 'field_interaction_type', 'use_lr', 'output_dim']

    def __init__(self, config: Config):
        super().__init__(model_name='FmFM', config=config)

        self.embed = Embeddings(config)
        self.interact_dim = int(config.num_fields * (config.num_fields - 1) / 2)
        if config.field_interaction_type == 'vectorized':
            self.interaction_weight = nn.Parameter(torch.Tensor(self.interact_dim, config.embed_size))
        elif config.field_interaction_type == 'matrixed':
            self.interaction_weight = nn.Parameter(torch.Tensor(self.interact_dim, config.embed_size, config.embed_size))
        nn.init.xavier_normal_(self.interaction_weight)
        self.lr_layer = LR(config) if config.use_lr else None
        self.upper_triange_mask = torch.triu(torch.ones(config.num_fields, config.num_fields - 1), 0).byte().to(config.device)
        self.lower_triange_mask = torch.tril(torch.ones(config.num_fields, config.num_fields - 1), -1).byte().to(config.device)
        if config.pretrain:
            self.feat_encoder = nn.Linear(self.interact_dim * config.embed_size, config.proj_size)
            self.criterion = IndexLinear(config)

    def forward(self, input_ids, labels=None, masked_index=None):
        feat_embed = self.embed(input_ids)
        field_wise_emb = feat_embed.unsqueeze(2).expand(-1, -1, self.config.num_fields - 1, -1)
        upper_tensor = torch.masked_select(field_wise_emb, self.upper_triange_mask.unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.config.embed_size)
        if self.config.field_interaction_type == 'vectorized':
            upper_tensor = upper_tensor * self.interaction_weight
        elif self.config.field_interaction_type == 'matrixed':
            upper_tensor = torch.matmul(upper_tensor.unsqueeze(2), self.interaction_weight).squeeze(2)
        lower_tensor = torch.masked_select(field_wise_emb.transpose(1, 2), self.lower_triange_mask.t().unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.config.embed_size)

        if self.config.pretrain:
            # Pretrain phase
            enc_output = self.feat_encoder((upper_tensor * lower_tensor).flatten(start_dim=1))
            selected_output = enc_output.unsqueeze(1).repeat(1, masked_index.shape[1], 1)
            loss, G_logits, G_features = self.criterion(labels, selected_output)
            total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
            outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        else:
            logits = (upper_tensor * lower_tensor).flatten(start_dim=1).sum(dim=-1, keepdim=True)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            outputs = self.get_outputs(logits, labels)
        return outputs


class PNN(BaseModel):
    used_params = ['embed_size', 'product_type', 'outer_product_kernel_type', 'output_dim', 
                   'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act']

    def __init__(self, config: Config):
        super().__init__(model_name='PNN', config=config)

        self.embed = Embeddings(config)
        input_dim = config.num_fields * (config.num_fields - 1) // 2 + config.num_fields * config.embed_size
        if config.product_type.lower() == 'inner':
            self.product_layer = InnerProductLayer(config.num_fields, output='inner_product')
        elif config.product_type.lower() == 'outer':
            self.product_layer = OuterProductLayer(config.num_fields, config.embed_size, config.outer_product_kernel_type)
        else:
            raise NotImplementedError

        if config.pretrain:
            self.feat_encoder = nn.Linear(input_dim, config.proj_size)
            self.criterion = IndexLinear(config)
        else:
            if config.num_hidden_layers > 0:
                self.dnn = MLPBlock(input_dim=input_dim,
                                    hidden_size=config.hidden_size,
                                    num_hidden_layers=config.num_hidden_layers,
                                    hidden_dropout_rate=config.hidden_dropout_rate,
                                    hidden_act=config.hidden_act)
                self.fc_out = nn.Linear(config.hidden_size, 1)
            else:
                self.dnn = None
                self.fc_out = nn.Linear(input_dim, 1)
            
    def forward(self, input_ids, labels=None, masked_index=None):
        feat_embed = self.embed(input_ids)
        product_vec = self.product_layer(feat_embed)
        dense_input = torch.cat([feat_embed.flatten(start_dim=1), product_vec], dim=1)
        
        if self.config.pretrain:
            # Pretrain phase
            enc_output = self.feat_encoder(dense_input)
            selected_output = enc_output.unsqueeze(1).repeat(1, masked_index.shape[1], 1)
            loss, G_logits, G_features = self.criterion(labels, selected_output)
            total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
            outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        else:
            if self.dnn is not None:
                nn_output = self.dnn(dense_input)
                logits = self.fc_out(nn_output)
            else:
                logits = self.fc_out(dense_input)
            outputs = self.get_outputs(logits, labels)
            
        return outputs


class DeepFM(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act', 'output_dim']

    def __init__(self, config: Config):
        super(DeepFM, self).__init__(model_name='DeepFM', config=config)
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
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
            self.dnn_fc_out = nn.Linear(config.hidden_size, config.output_dim)
        # self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data = torch.zeros(m.bias.data.shape)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        dnn_vec = self.dnn(feat_embed.flatten(start_dim=1))
        
        if self.config.pretrain:
            lr_fm = self.lr_layer(input_ids)[0] + self.ip_layer(feat_embed)
            final_vec = torch.cat([dnn_vec, lr_fm], dim=1)
            outputs = self.get_pretraining_output(final_vec, labels, masked_index)
        else:
            logits = self.dnn_fc_out(dnn_vec)
            logits += self.lr_layer(input_ids)[0]
            logits += self.ip_layer(feat_embed)
            outputs = self.get_outputs(logits, labels)

        return outputs


class xDeepFM(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'cin_layer_units', 'use_lr', 'output_dim']

    def __init__(self, config: Config):
        super(xDeepFM, self).__init__(model_name='xDeepFM', config=config)
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
            self.embed = Embeddings(config)

        input_dim = config.num_fields * config.embed_size
        cin_layer_units = [int(c) for c in config.cin_layer_units.split(',')]
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
            self.fc = nn.Linear(final_dim, config.output_dim)

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        final_vec = self.cin(feat_embed)
        if self.dnn is not None:
            dnn_vec = self.dnn(feat_embed.flatten(start_dim=1))
            final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(final_vec, labels, masked_index)
        else:
            logits = self.fc(final_vec)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            outputs = self.get_outputs(logits, labels)

        return outputs


class DCNV2(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'num_cross_layers', 'output_dim']

    def __init__(self, config: Config):
        super(DCNV2, self).__init__(model_name='DCNV2', config=config)
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
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
            self.fc_out = nn.Linear(final_dim, config.output_dim)

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids).flatten(start_dim=1)
        cross_output = self.cross_net(feat_embed)
        if self.config.num_hidden_layers > 0:
            dnn_output = self.parallel_dnn(feat_embed)
            final_output = torch.cat([cross_output, dnn_output], dim=-1)
        else:
            final_output = cross_output
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(final_output, labels, masked_index)
        else:
            logits = self.fc_out(final_output)
            outputs = self.get_outputs(logits, labels)

        return outputs


class FGCNN(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'share_embedding', 'channels', 'kernel_heights', 'pooling_sizes', 'recombined_channels',
                   'conv_act', 'output_dim']

    def __init__(self, config: Config):
        super(FGCNN, self).__init__(model_name='fgcnn', config=config)
        
        # No need for the embedding layer for the generator of RFD
        self.share_embedding = config.share_embedding
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
            self.fg_embed = None
        else:
            self.embed = Embeddings(config)
            if not self.share_embedding:
                self.fg_embed = Embeddings(config)
        channels = [int(c) for c in config.channels.split(',')]
        kernel_heights = [int(c) for c in config.kernel_heights.split(',')]
        pooling_sizes = [int(c) for c in config.pooling_sizes.split(',')]
        recombined_channels = [int(c) for c in config.recombined_channels.split(',')]
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
        self.ip_layer = InnerProductLayer(total_features, output='inner_product')
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

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None, feat_embed2=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None: # Not the generator of RFD
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
            outputs = self.get_pretraining_output(dense_input, labels, masked_index)
        else:
            if self.dnn is not None:
                nn_output = self.dnn(dense_input)
                logits = self.fc_out(nn_output)
            else:
                logits = self.fc_out(dense_input)
            outputs = self.get_outputs(logits, labels)

        return outputs


class FiBiNet(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'use_lr', 'reduction_ratio', 'bilinear_type', 'output_dim']

    def __init__(self, config: Config):
        super(FiBiNet, self).__init__(model_name='FiBiNet', config=config)
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
            self.embed = Embeddings(config)
        
        self.senet_layer = SqueezeExtractionLayer(config)
        self.bilinear_layer = BilinearInteractionLayer(config)
        self.lr_layer = LR(config) if config.use_lr else None
        final_dim = config.num_fields * (config.num_fields - 1) * config.embed_size
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

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        senet_embed = self.senet_layer(feat_embed)
        bilinear_p = self.bilinear_layer(feat_embed)
        bilinear_q = self.bilinear_layer(senet_embed)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(comb_out, labels, masked_index)
        else:
            if self.dnn is not None:
                logits = self.fc_out(self.dnn(comb_out))
            else:
                logits = self.fc_out(comb_out)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            outputs = self.get_outputs(logits, labels)
        
        return outputs


class FiGNN(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'res_conn', 'reuse_graph_layer', 'output_dim']

    def __init__(self, config: Config):
        super(FiGNN, self).__init__(model_name='FiGNN', config=config)
        logger.warning('this model requires embed_size == hidden_size, only uses embed_size')
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
            self.embed = Embeddings(config)
        self.fignn = FiGNNBlock(config)

        if config.pretrain:
            final_dim = config.num_fields * config.embed_size
            self.create_pretraining_predictor(final_dim)
        else:
            self.fc = AttentionalPrediction(config)

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        h_out = self.fignn(feat_embed)
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(h_out.flatten(start_dim=1), labels, masked_index)
        else:
            logits = self.fc(h_out)
            outputs = self.get_outputs(logits, labels)
        
        return outputs


class AutoInt(BaseModel):
    used_params = ['embed_size', 'num_attn_layers', 'attn_size', 'num_attn_heads', 
                   'attn_probs_dropout_rate', 'use_lr', 'res_conn', 'attn_scale', 'output_dim', 
                   'dnn_size', 'num_dnn_layers', 'dnn_act', 'dnn_drop']

    def __init__(self, config: Config):
        super(AutoInt, self).__init__(model_name='AutoInt', config=config)
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
            self.embed = Embeddings(config)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(config.embed_size if i == 0 else config.num_attn_heads * config.attn_size,
                                     attention_dim=config.attn_size, 
                                     num_heads=config.num_attn_heads, 
                                     dropout_rate=config.attn_probs_dropout_rate, 
                                     use_residual=config.res_conn, 
                                     use_scale=config.attn_scale, 
                                     layer_norm=False,
                                     align_to='output') 
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
            
    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        attention_out = self.self_attention(feat_embed)
        attention_out = torch.flatten(attention_out, start_dim=1)
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(attention_out, labels, masked_index)
        else:
            logits = self.attn_out(attention_out)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            if self.dnn is not None:
                logits += self.dnn_out(self.dnn(feat_embed.flatten(start_dim=1)))
            outputs = self.get_outputs(logits, labels)
        return outputs


class InterHAt(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers',
                   'attn_size', 'num_attn_heads', 'attn_probs_dropout_rate', 'res_conn', 'output_dim', 
                   'dnn_size', 'num_dnn_layers', 'dnn_act', 'dnn_drop']

    def __init__(self, config: Config):
        super(InterHAt, self).__init__(model_name='InterHAt', config=config)
        self.embed = Embeddings(config)
        self.order = config.num_hidden_layers
        self.multi_head_attention = InterHAt_MultiHeadSelfAttention(config.embed_size, 
                                                                    config.attn_size, 
                                                                    config.num_attn_heads,
                                                                    dropout_rate=config.attn_probs_dropout_rate,
                                                                    use_residual=config.res_conn,
                                                                    use_scale=True,
                                                                    layer_norm=True)
        self.feedforward = InterHAt_FeedForwardNetwork(config.embed_size, 
                                                       hidden_dim=config.hidden_size,
                                                       layer_norm=True, 
                                                       use_residual=config.res_conn)
        self.aggregation_layers = nn.ModuleList([InterHAt_AttentionalAggregation(config.embed_size, config.hidden_size) 
                                                 for _ in range(self.order)])
        self.attentional_score = InterHAt_AttentionalAggregation(config.embed_size, config.hidden_size)

        if config.pretrain:
            self.feat_encoder = nn.Linear(config.embed_size, config.proj_size)
            self.criterion = IndexLinear(config)
        else:
            if config.num_dnn_layers > 0:
                self.mlp = MLPBlock(input_dim=config.embed_size,
                                    hidden_size=config.dnn_size,
                                    num_hidden_layers=config.num_dnn_layers,
                                    hidden_dropout_rate=config.dnn_drop,
                                    hidden_act=config.dnn_act)
                self.fc_out = nn.Linear(config.dnn_size, config.output_dim)
            else:
                self.mlp = None
                self.fc_out = nn.Linear(config.embed_size, config.output_dim)
            
    def forward(self, input_ids, labels=None, masked_index=None):
        X0 = self.embed(input_ids)
        X1 = self.feedforward(self.multi_head_attention(X0))
        X_p = X1
        agg_u = []
        for p in range(self.order):
            u_p = self.aggregation_layers[p](X_p) # b x emb
            agg_u.append(u_p)
            if p != self.order - 1:
                X_p = u_p.unsqueeze(1) * X1 + X_p
        U = torch.stack(agg_u, dim=1) # b x order x emb
        u_f = self.attentional_score(U)

        if self.config.pretrain:
            # Pretrain phase
            enc_output = self.feat_encoder(u_f)
            selected_output = enc_output.unsqueeze(1).repeat(1, masked_index.shape[1], 1)
            loss, G_logits, G_features = self.criterion(labels, selected_output)
            total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
            outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        else:
            if self.mlp is None:
                logits = self.fc_out(u_f)
            else:
                logits = self.fc_out(self.mlp(u_f))
            outputs = self.get_outputs(logits, labels)
        return outputs


class Transformer(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'num_attn_heads', 'intermediate_size', 'output_reduction', 
                   'norm_first', 'layer_norm_eps', 'use_lr', 'output_dim', 
                   'dnn_size', 'num_dnn_layers', 'dnn_act', 'dnn_drop']

    def __init__(self, config: Config):
        super(Transformer, self).__init__(model_name='trans', config=config)
        logger.warning('this model requires embed_size == hidden_size, only uses embed_size')
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == 'RFD' and config.is_generator:
            self.embed = None
        else:
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
            if self.config.output_reduction == 'fc':
                self.trans_out = nn.Linear(config.num_fields * config.embed_size, config.output_dim)
            elif self.config.output_reduction == 'mean,fc' or self.config.output_reduction == 'sum,fc':
                self.trans_out = nn.Linear(config.embed_size, config.output_dim)
            elif self.config.output_reduction == 'attn,fc':
                self.field_reduction_attn = nn.Sequential(
                    nn.Linear(config.embed_size, config.embed_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.embed_size, 1),
                    nn.Softmax(dim=1),
                )
                self.trans_out = nn.Linear(config.embed_size, config.output_dim)
            else:
                raise NotImplementedError
            self.lr_layer = LR(config) if config.use_lr else None
            if config.num_dnn_layers > 0:
                self.mlp = MLPBlock(input_dim=config.num_fields * config.embed_size,
                                    hidden_size=config.dnn_size,
                                    num_hidden_layers=config.num_dnn_layers,
                                    hidden_dropout_rate=config.dnn_drop,
                                    hidden_act=config.dnn_act)
                self.mlp_out = nn.Linear(config.dnn_size, config.output_dim)
            else:
                self.mlp = None

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        enc_output = self.encoder(feat_embed)
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(enc_output.flatten(start_dim=1), labels, masked_index)
        else:
            # Finetune or train from scratch
            if self.config.output_reduction == 'fc':
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == 'mean,fc':
                enc_output = torch.sum(enc_output, dim=1) / self.config.num_fields
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == 'sum,fc':
                enc_output = torch.sum(enc_output, dim=1)
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == 'attn,fc':
                attn_score = self.field_reduction_attn(enc_output)
                attn_feat = torch.sum(enc_output * attn_score, dim=1)
                logits = self.trans_out(attn_feat)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            if self.mlp is not None:
                logits += self.mlp_out(self.mlp(feat_embed.flatten(start_dim=1)))
            outputs = self.get_outputs(logits, labels)
        return outputs


class PUNV4(BaseModel):
    used_params = ['embed_size', 'hidden_size', 'num_hidden_layers', 'hidden_dropout_rate', 'hidden_act',
                   'num_attn_heads', 'attn_probs_dropout_rate', 'intermediate_size', 'inter_layer_norm',
                   'norm_first', 'layer_norm_eps', 'output_dim']

    def __init__(self, config: Config):
        super(PUNV4, self).__init__(model_name='PUNV4', config=config)
        logger.warning('this model requires embed_size == hidden_size, only uses embed_size')
        self.pun_embed = config.pun_embed
        if self.pun_embed:
            self.embed = PUNEmbeddings(config)
        else:
            self.embed = Embeddings(config)
        prod_layers = []
        for i in range(config.num_hidden_layers):
            prod_layers.append(ProductLayerV4(config))
        self.encoder = nn.Sequential(*prod_layers)
        self.fc_out = nn.Linear(config.num_fields * config.hidden_size, config.output_dim)

    def forward(self, input_ids, field_ids, type_ids, labels=None):
        if self.pun_embed:
            feat_embed = self.embed(input_ids, field_ids, type_ids)
        else:
            feat_embed = self.embed(input_ids)
        # feat_embed = self.embed(input_ids)
        enc_output = self.encoder(feat_embed)
        logits = self.fc_out(enc_output.flatten(start_dim=1))

        outputs = self.get_outputs(logits, labels)
        return outputs


class PUN(BaseModel):
    used_params = ['pun_embed', 'embed_size', 'embed_norm', 'hidden_size', 'num_hidden_layers', 'num_channels',
                   'num_attn_heads', 'attn_probs_dropout_rate', 'agg_type',
                   'prod_layer_norm', 'layer_norm_eps', 'norm_first', 'res_conn',
                   'hidden_dropout_rate', 'hidden_act', 'output_reduction', 'output_dim']

    def __init__(self, config: Config):
        super(PUN, self).__init__(model_name='PUN', config=config)
        logger.warning('this model requires embed_size == hidden_size, only uses embed_size')
        self.pun_embed = config.pun_embed
        if self.pun_embed:
            self.embed = PUNEmbeddings(config)
        else:
            self.embed = Embeddings(config)
        prod_layers = []
        c_in = 1
        for i in range(config.num_hidden_layers):
            prod_layers.append(ProductLayer(config, c_in=c_in, c_out=config.num_channels))
            prod_layers.append(get_act(config.hidden_act))
            prod_layers.append(nn.Dropout(p=config.hidden_dropout_rate))
            c_in = config.num_channels
        # batch * num_fields * c_out * embed
        self.prod_net = nn.Sequential(*prod_layers)
        self.cls = MultiChannelOutputHead(config)

    def forward(self, input_ids, field_ids, type_ids, labels=None):
        # batch * num_fields * 1 * embed
        if self.pun_embed:
            feat_embed = self.embed(input_ids, field_ids, type_ids).unsqueeze(dim=2)
        else:
            feat_embed = self.embed(input_ids).unsqueeze(dim=2)
        # batch * num_fields * c_out * embed
        nn_output = self.prod_net(feat_embed)
        logits = self.cls(nn_output)

        outputs = self.get_outputs(logits, labels)
        return outputs


class PUNV5(BaseModel):
    used_params = ['pun_embed', 'embed_size', 'embed_norm', 'hidden_size', 'num_hidden_layers', 'num_channels',
                   'num_attn_heads', 'attn_probs_dropout_rate', 'agg_type',
                   'prod_layer_norm', 'layer_norm_eps', 'norm_first', 'res_conn', 'prod_dropout_rate',
                   'intermediate_size', 'inter_layer_norm', 'hidden_dropout_rate', 'hidden_act',
                   'pt_neg_num', 'pun_pt_channel_agg', 'output_reduction', 'output_dim', 
                   'use_lr', 'dnn_size', 'num_dnn_layers', 'dnn_act', 'dnn_drop']

    def __init__(self, config: Config):
        super(PUNV5, self).__init__(model_name='PUNV5', config=config)
        logger.warning('this model requires embed_size == hidden_size, only uses embed_size')
        self.pun_embed = config.pun_embed
        self.embeddings = PUNEmbeddings(config) if self.pun_embed else Embeddings(config)
        prod_layers = []
        c_in = 1
        for i in range(config.num_hidden_layers):
            prod_layers.append(ProductLayerV5(config, c_in=c_in, c_out=config.num_channels))
            c_in = config.num_channels
        self.encoder = nn.Sequential(*prod_layers)
        if config.pretrain:
            if self.config.pun_pt_channel_agg == 'sum' or self.config.pun_pt_channel_agg == 'mean':
                self.criterion = IndexLinear(config)
        else:
            self.cls = MultiChannelOutputHead(config)
            self.lr_layer = LR(config) if config.use_lr else None
            if config.num_dnn_layers > 0:
                self.mlp = MLPBlock(input_dim=config.num_fields * config.embed_size,
                                    hidden_size=config.dnn_size,
                                    num_hidden_layers=config.num_dnn_layers,
                                    hidden_dropout_rate=config.dnn_drop,
                                    hidden_act=config.dnn_act)
                self.mlp_out = nn.Linear(config.dnn_size, config.output_dim)
            else:
                self.mlp = None

    def forward(self, input_ids, field_ids, type_ids, labels=None, masked_index=None):
        embedding_output = self.embeddings(input_ids, field_ids, type_ids) if self.pun_embed else \
            self.embeddings(input_ids)
        encoder_output = self.encoder(embedding_output.unsqueeze(dim=2))
        if self.config.pretrain:
            if self.config.pun_pt_channel_agg == 'sum':
                enc_output = torch.sum(encoder_output, dim=2)
            elif self.config.pun_pt_channel_agg == 'mean':
                enc_output = torch.sum(encoder_output, dim=2) / self.config.num_channels
            selected_output = torch.gather(enc_output, 1, masked_index.unsqueeze(-1).repeat(1, 1, enc_output.shape[2]))
            loss, G_logits, G_features = self.criterion(labels, selected_output)
            total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
            outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        else:
            logits = self.cls(encoder_output)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            if self.mlp is not None:
                logits += self.mlp_out(self.mlp(embedding_output.flatten(start_dim=1)))
            outputs = self.get_outputs(logits, labels)
        return outputs

    def load_for_finetune(self, model_path):
        pretrained_dict = torch.load(model_path)
        # print(len(pretrained_dict))
        model_dict = self.state_dict()
        count = 0
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape:
                model_dict[k] = v
            else:
                # print(k)
                count += 1
        self.load_state_dict(model_dict)


class PUNV6(BaseModel):
    used_params = ['pun_embed', 'embed_size', 'embed_norm', 'hidden_size', 'num_hidden_layers', 'num_channels',
                   'num_attn_heads', 'attn_probs_dropout_rate', 'agg_type',
                   'prod_layer_norm', 'layer_norm_eps', 'norm_first', 'res_conn', 'prod_dropout_rate',
                   'intermediate_size', 'inter_layer_norm', 'hidden_dropout_rate', 'hidden_act',
                   'output_reduction', 'output_dim']

    def __init__(self, config: Config):
        super(PUNV6, self).__init__(model_name='PUNV6', config=config)
        self.pun_embed = config.pun_embed
        self.embeddings = PUNEmbeddings(config) if self.pun_embed else Embeddings(config)
        prod_layers = []
        c_in = 1
        for i in range(config.num_hidden_layers):
            prod_layers.append(ProductLayerV6(config, c_in=c_in, c_out=config.num_channels))
            c_in = config.num_channels
        self.encoder = nn.Sequential(*prod_layers)
        self.cls = MultiChannelOutputHead(config)

    def forward(self, input_ids, field_ids, type_ids, labels=None):
        embedding_output = self.embeddings(input_ids, field_ids, type_ids) if self.pun_embed else \
            self.embeddings(input_ids)
        encoder_output = self.encoder(embedding_output.unsqueeze(dim=2))
        logits = self.cls(encoder_output)

        outputs = self.get_outputs(logits, labels)
        return outputs


def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def replaced_feature_sample(G_logits):
    F.Softmax(G_logits, dim=-1)

class RFD(nn.Module):
    def __init__(self, generator: BaseModel, discriminator: BaseModel, config: Config):
        super().__init__()
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        if self.config.RFD_G in ['Unigram', 'Uniform', 'Whole-Unigram', 'Whole-Uniform']:
            self.reserved_zero = torch.Tensor([0.0])

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and 'embedding' in name:
                print(f'{name}: IGNORE')
                continue
            if param.requires_grad:
                total_params += param.numel()
                print(f'{name}: {param.shape}')
        logger.info(f'total number of parameters: {total_params}')
        return total_params

    def forward(self, input_ids, labels=None, masked_index=None, origin_input_ids=None):
        # Run the generator
        if self.config.RFD_G == 'Model':
            g_feat_embed = self.discriminator.embed(input_ids) # We share the embedding for G & D.
            G_loss, G_count, G_logits, G_features = self.generator(input_ids, labels, masked_index, g_feat_embed)
            # Sample and replace features according to G, and construct the input for D.
            sampled = gumbel_sample(G_logits, temperature=self.config.gumbel_temp)
            # sampled = replaced_feature_sample(G_logits)
            sampled_features = torch.gather(G_features, 2, sampled.unsqueeze(2)).squeeze(2)
            disc_input_ids = torch.scatter(origin_input_ids, 1, masked_index, sampled_features.detach())
            disc_labels = (origin_input_ids != disc_input_ids).float().detach()
        elif self.config.RFD_G in ['Unigram', 'Uniform', 'Whole-Unigram', 'Whole-Uniform']:
            disc_input_ids = input_ids
            disc_labels = labels
        else:
            raise NotImplementedError

        # Run the discriminator
        D_loss, D_count, D_logits = self.discriminator(disc_input_ids, disc_labels)

        # Combine the loss
        loss = self.config.D_w * D_loss
        if self.config.RFD_G == 'Model':
            loss += self.config.G_w * G_loss
            
        # Compute the accuracy metric for G & D
        with torch.no_grad():
            if self.config.RFD_G == 'Model':
                G_acc = (G_logits.argmax(dim=2) == 0).sum() / G_count
            else:
                G_loss, G_acc, G_count = self.reserved_zero, self.reserved_zero, 1
            D_acc = ((torch.sigmoid(D_logits) > 0.5).float() == disc_labels).sum() / D_count
            D_input_pos_ratio = disc_labels.mean()

        outputs = (loss, G_loss, G_acc, G_count, D_loss, D_acc, D_count, D_input_pos_ratio)

        return outputs