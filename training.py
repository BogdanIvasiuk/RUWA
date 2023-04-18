from torch.utils.data import RandomSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange, tqdm_notebook
from tensorboardX import SummaryWriter
import math
import os
import torch

class TrainingSpace:
  def __init__(self, args):
    self.args = args
    self.tb_writer = SummaryWriter()

  def train(self, train_dataset, model, tokenizer,logger):
    """
    Trains the model with training dataset
    """
    # Initialize various necessary objects
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args['train_batch_size'])

    t_total = self.compute_training_time(train_dataloader)
    optimizer_grouped_parameters = self.get_optimizer_params(model)
    optimizer, scheduler = self.initialize_optimizer_and_scheduler(optimizer_grouped_parameters, t_total)

    global_step, tr_loss, model= self.train_model(train_dataloader, model, optimizer, scheduler, self.tb_writer,logger)

    return global_step, tr_loss / global_step, model

  def compute_training_time(self, dataloader):
    # Compute the total time
    t_total = len(dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']
    warmup_steps = math.ceil(t_total * self.args['warmup_ratio'])
    self.args['warmup_steps'] = warmup_steps if self.args['warmup_steps'] == 0 else self.args['warmup_steps']
    return t_total

  def get_optimizer_params(self, model):
    # Set the grouped parameters for optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters

  def initialize_optimizer_and_scheduler(self, optimizer_grouped_parameters, t_total):
    # Initialize optimizer as Adam with constant weight decay and a linear scheduler with warmup
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args['warmup_steps'], num_training_steps=t_total)
    return optimizer, scheduler
  

  def load_model_from_checkpoint(self, checkpoint_folder, model):
    # Carica il modello e il tokenizer dalla cartella specificata
    model = model.from_pretrained(checkpoint_folder)

    # Carica le variabili global_step e tr_loss dal file corrispondente
    global_step = torch.load(os.path.join(checkpoint_folder, "global_step.pt"))
    tr_loss = torch.load(os.path.join(checkpoint_folder, "training_loss.pt"))

    return global_step, tr_loss, model

  def train_model(self, dataloader, model, optimizer, scheduler, tb_writer, logger):
    
    # Check if model has already been trained and saved
    if os.path.exists(self.args['output_dir']):
        # Trova il checkpoint più recente nella cartella
        checkpoint_dir = max([os.path.join(self.args['output_dir'], f) for f in os.listdir(self.args['output_dir']) if 'checkpoint-' in f], key=os.path.getctime)
        checkpoints = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f)) and f.endswith('.bin')]
        if len(checkpoints) > 0:
            logger.info(f"Model has already been trained and saved in {self.args['output_dir']}")
  
            return self.load_model_from_checkpoint(checkpoint_dir, model)
        
    # Initialize variables for training
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(self.args['num_train_epochs']), desc="Epoch")

    # Start training!
    for epoch in train_iterator:
      epoch_iterator = tqdm_notebook(dataloader, desc="Iteration")
      for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to(model.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3]}
        outputs = model(**inputs)
        loss = outputs[0]

        if self.args['gradient_accumulation_steps'] > 1:
          loss = loss / self.args['gradient_accumulation_steps']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args['max_grad_norm'])

        tr_loss += loss.item()
        if (step + 1) % self.args['gradient_accumulation_steps'] == 0:
          optimizer.step()
          scheduler.step()
          model.zero_grad()
          global_step += 1

          if self.args['logging_steps'] > 0 and global_step % self.args['logging_steps'] == 0:
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/self.args['logging_steps'], global_step)
            logging_loss = tr_loss

          if self.args['save_steps'] > 0 and global_step % self.args['save_steps'] == 0:
            output_dir = os.path.join(self.args['output_dir'], 'checkpoint-{}'.format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            torch.save(tr_loss, os.path.join(output_dir, "training_loss.pt"))
            torch.save(global_step, os.path.join(output_dir, "global_step.pt")) # Salva global_step

      # Log epoch summary
      epoch_loss = tr_loss / global_step
      tb_writer.add_scalar('epoch_loss', epoch_loss, epoch)
      train_iterator.set_postfix(loss=epoch_loss)

    return global_step, tr_loss, model
