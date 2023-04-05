import os
import torch
from config import MODEL_CLASSES, args, device, train_articles, train_SI_labels, logger

config_class, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
config = config_class.from_pretrained(args["model_name"], num_labels=2, finetuning_task="binary")
tokenizer = tokenizer_class.from_pretrained(args["model_name"])
model_prop = model_class(config)
model_prop.to(device)

train_dataset, df_train = generate_training_dataset_from_articles([train_articles], [train_SI_labels], tokenizer)
global_step, tr_loss = train(train_dataset, model_prop, tokenizer)
logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
logger.info("Saving model checkpoint to %s", args['output_dir'])
model_to_save = model_prop.module if hasattr(model_prop, 'module') else model_prop
model_to_save.save_pretrained(args['output_dir'])
tokenizer.save_pretrained(args['output_dir'])
torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))