import logging
import os

from config import Config, ModelArguments
from trainer import Trainer
# from pyx_utils.trainer import Trainer
from models import BaseModel, RFD
from training_args import TrainingArguments
from hf_argparser import HfArgumentParser
from utils import set_seed, setup_print_for_ddp
from dataset import BaseDataset, DataCollatorV1, DataCollatorV2


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    training_log = os.path.join(training_args.output_dir, 'train.log')
    results_log = os.path.join(training_args.output_dir, 'results.log')
    if os.path.exists(results_log):
        print('job already finished, quit')
        exit(0)

    # Setup logging
    setup_print_for_ddp((training_args.local_rank in [-1, 0]))
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        # filename=f'{training_args.output_dir}/log',
        # filemode='w',
    )
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(filename=training_log, mode='w'))
    logger.warning(f'Process rank: {training_args.local_rank}, device: {training_args.device}, '
                   f'n_gpu: {training_args.n_gpu}, distributed_training: {bool(training_args.local_rank != -1)}, '
                   f'16-bits training: {training_args.fp16}')
    logger.info(f'training/evaluation parameters {training_args}')

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    dataset = BaseDataset.get_dataset(training_args)
    if 'pun' in model_args.model_name.lower():
        if training_args.dataset_name == 'avazu_demo':
            train_dataset, eval_dataset = dataset.get_dataset_v2(eval=False), dataset.get_dataset_v2(eval=True)
        else:
            train_dataset, eval_dataset, test_dataset = \
                dataset.get_dataset_v2('train'), dataset.get_dataset_v2('valid'), dataset.get_dataset_v2('test')
        data_collator = DataCollatorV2
    else:
        train_dataset, eval_dataset, test_dataset = \
            dataset.get_dataset_v1('train'), dataset.get_dataset_v1('valid'), dataset.get_dataset_v1('test')
        data_collator = DataCollatorV1
    logger.info(f'field_names = {dataset.field_names}')
    logger.info(f'feat_map example:')
    for i, (k, v) in enumerate(dataset.feat_map.items()):
        logger.info(f'{k}\t{v}')
        if i == 10:
            break

    # Build model_config
    model_config_dict = model_args.to_dict()
    model_config_dict['data_dir'] = training_args.data_dir
    model_config_dict['input_size'] = len(dataset.feat_map)
    model_config_dict['num_fields'] = len(dataset.field_map) - 1 # There is an additional reserved field
    model_config_dict['num_feat_types'] = len(dataset.feat_type_map)
    model_config_dict['pretrain'] = training_args.pretrain
    model_config_dict['pt_type'] = training_args.pt_type
    model_config_dict['RFD_G'] = training_args.RFD_G
    model_config_dict['feat_count'] = dataset.feat_count
    model_config_dict['n_core'] = training_args.n_core
    model_config_dict['device'] = training_args.device
    model_config_dict['n_gpu'] = training_args.n_gpu
    model_config_dict['idx_low'] = dataset.idx_low
    model_config_dict['idx_high'] = dataset.idx_high
    model_config_dict['feat_num_per_field'] = dataset.feat_num_per_field

    # Build the model
    if training_args.pretrain and training_args.pt_type == 'RFD':
        # The generator for Replaced Feature Detection
        if training_args.RFD_G == 'Model':
            model_config_dict['is_generator'] = True
            g_config = Config.from_dict(model_config_dict)
            generator = BaseModel.from_config(g_config)
        elif training_args.RFD_G in ['Unigram', 'Uniform', 'Whole-Unigram', 'Whole-Uniform']:
            generator = None
        else:
            raise NotImplementedError
        # The discriminator for Replaced Feature Detection
        model_config_dict['is_generator'] = False
        d_config = Config.from_dict(model_config_dict)
        discriminator = BaseModel.from_config(d_config)
        # The packaging model
        model_config_dict['is_generator'] = None
        config = Config.from_dict(model_config_dict)
        model = RFD(generator, discriminator, config)
    else:
        config = Config.from_dict(model_config_dict)
        model = BaseModel.from_config(config)
        if training_args.finetune:
            model.load_for_finetune(training_args.finetune_model_path)

    # Run the trainer
    trainer = Trainer(model, config, training_args, train_dataset=train_dataset,
                      eval_dataset=eval_dataset, data_collator=data_collator)
    if training_args.pretrain:
        if training_args.pt_type == 'MFP':
            trainer.MFP_pretrain()
        elif training_args.pt_type == 'RFD':
            trainer.RFD_pretrain()
        elif training_args.pt_type == 'SCARF':
            trainer.SCARF_pretrain()
        elif training_args.pt_type == 'MF4UIP':
            trainer.MF4UIP_pretrain()
        else:
            raise NotImplementedError
    else:
        trainer.train()
        trainer.test(test_dataset)
    
    # Write the result log
    lines = open(training_log, 'r').readlines()
    writer = open(results_log, 'w')
    writer.write(''.join(lines))
    writer.close()
