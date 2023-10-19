import logging
import os

from transformers import HfArgumentParser, set_seed

from arguments import Config, ModelArguments, TrainingArguments
from trainer import Trainer
# from pyx_utils.trainer import Trainer
from models import BaseModel
from dataset import BaseDataset


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    training_log = os.path.join(training_args.output_dir, "train.log")
    results_log = os.path.join(training_args.output_dir, "results.log")
    if os.path.exists(results_log):
        print("job already finished, quit")
        exit(0)

    # Setup logging
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        # filename=f"{training_args.output_dir}/log",
        # filemode="w",
    )
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(filename=training_log, mode="w"))
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed_training: {bool(training_args.local_rank != -1)}"
    )
    logger.info(f"training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    dataset = BaseDataset(training_args)
    datasets = {split: dataset.get_splited_dataset(split) for split in dataset.split_names}
    logger.info(f"field_names = {dataset.field_names}")

    # Build model_config
    model_config_dict = model_args.to_dict()
    model_config_dict["data_dir"] = training_args.data_dir
    model_config_dict["input_size"] = len(dataset.feat_map)
    model_config_dict["num_fields"] = len(dataset.field_map) - 1 # NOTE: There is an additional reserved field in our preprocessed data
    model_config_dict["pretrain"] = training_args.pretrain
    model_config_dict["pt_type"] = training_args.pt_type
    model_config_dict["RFD_replace"] = training_args.RFD_replace
    model_config_dict["feat_count"] = dataset.feat_count
    model_config_dict["device"] = training_args.device
    model_config_dict["n_gpu"] = training_args.n_gpu
    model_config_dict["idx_low"] = dataset.idx_low
    model_config_dict["idx_high"] = dataset.idx_high
    model_config_dict["feat_num_per_field"] = dataset.feat_num_per_field

    # Build the model
    config = Config.from_dict(model_config_dict)
    model = BaseModel.from_config(config)
    if training_args.finetune:
        model.load_for_finetune(training_args.pretrained_model_path)
    
    # Run the trainer
    trainer = Trainer(
        model, 
        config, 
        training_args, 
        train_dataset=datasets["train"], 
        eval_dataset=datasets["valid"], 
    )
    if training_args.pretrain:
        if training_args.pt_type == "MFP":
            trainer.MFP_pretrain()
        elif training_args.pt_type == "RFD":
            trainer.RFD_pretrain()
        else:
            raise NotImplementedError
    else:
        trainer.train()
        trainer.test(datasets["test"])
    
    # Write the result log
    lines = open(training_log, "r").readlines()
    writer = open(results_log, "w")
    writer.write("".join(lines))
    writer.close()
