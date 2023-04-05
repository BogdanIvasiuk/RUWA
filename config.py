import torch
import logging
from transformers import (BertConfig, BertTokenizer, BertForSequenceClassification, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)

train_articles = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/Propaganda/PTC_CORPUS/datasets/train-articles"
dev_articles = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/Propaganda/PTC_CORPUS/datasets/dev-articles"
train_SI_labels = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/Propaganda/PTC_CORPUS/datasets/train-labels-task-si"
train_TC_labels = ""
dev_SI_labels = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/Propaganda/PTC_CORPUS/datasets/dev-labels-task-si"
dev_TC_labels = ""
dev_TC_labels_file = ""
dev_TC_template = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/Propaganda/PTC_CORPUS/datasets/dev-task-TC-template.out"
techniques = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/Propaganda/PTC_CORPUS/tools/data/propaganda-techniques-names-semeval2020task11.txt"
PROP_TECH_TO_LABEL = {}
LABEL_TO_PROP_TECH = {}
label = 0
with open(techniques, "r") as f:
  for technique in f:
    PROP_TECH_TO_LABEL[technique.replace("\n", "")] = int(label)
    LABEL_TO_PROP_TECH[int(label)] = technique.replace("\n", "")
    label += 1
device = torch.device("cuda") #cpu o cuda
n_gpu = torch.cuda.device_count()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LOG")
MODEL_CLASSES = {"bert": (BertConfig, BertForSequenceClassification, BertTokenizer)}
args = {"data_dir": "datasets/",
        "model_type": "bert",
        "model_name": "bert-base-uncased",
        "output_dir": "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/Propaganda/Task_SI/",
        "max_seq_length": 128,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_train_epochs": 1,
        "weight_decay": 0,
        "learning_rate": 4e-5,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "logging_steps": 50,
        "save_steps": 2000,
        "overwrite_output_dir": False}