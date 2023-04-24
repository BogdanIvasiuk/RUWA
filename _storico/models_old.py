from info import MODEL_CLASSES
import torch
import math
from torch.utils.data import RandomSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange, tqdm_notebook
import tqdm
from tensorboardX import SummaryWriter
import os


class ModelPropaganda(torch.nn.Module):
    def __init__(self, args):
        super(ModelPropaganda, self).__init__()
        self.args = args
        
        # Call methods that will load the tokenizer and the model
        self.tokenizer = self.load_tokenizer()
        self.config = self.configuration()
        self.model = self.load_model()

        
    def configuration(self):
        """Load configuration for the model"""
        # Get the tuple containing necessary information from the MODEL_CLASSES dictionary
        config_class, _, _ = MODEL_CLASSES[self.args['model_type']]
        # Create an instance of the configuration class using the model_name argument in self.args
        return config_class.from_pretrained(self.args['model_name'], num_labels=self.args['num_labels'], finetuning_task=self.args['finetuning_task'])
    
    def load_tokenizer(self):
        """Load the tokenizer"""
        # Get the tuple containing necessary information from the MODEL_CLASSES dictionary
        _, _, tokenizer_class = MODEL_CLASSES[self.args['model_type']]
        # Create an instance of the tokenizer using the model_name argument in self.args
        return tokenizer_class.from_pretrained(self.args['model_name'])
      
    def load_model(self):
        """Load the model"""
        # Get the tuple containing necessary information from the MODEL_CLASSES dictionary
        _, model_class, _ = MODEL_CLASSES[self.args['model_type']]
        # Create an instance of the model using the configuration retrieved in 'configuration'
        model = model_class.from_pretrained(self.args['model_name'], config=self.config, cache_dir=None)
        # Move the model to the GPU if available
        device = torch.device(self.args["device"]) if torch.cuda.is_available() and self.args["device"] == "cuda" else torch.device("cpu")
        return model.to(device)
    
    def compute_training_time(self, dataloader):
        """
        Calculates the total number of training steps needed for the specified dataloader,
        based on the gradient accumulation steps and the number of epochs specified by the user.

        Args:
            dataloader: The PyTorch dataloader to be used for training.

        Returns:
            The total number of training steps required.
        """

        # Calculate the total number of steps required for training the model
        t_total = len(dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']
        
        # Calculate the number of warmup steps based on the warmup ratio
        # and the total number of steps. The warmup ratio is the percentage
        # of steps over which the learning rate will be gradually increased.
        warmup_steps = math.ceil(t_total * self.args['warmup_ratio'])
        
        # If the user has not specified a value for warmup steps,
        # set it to the value calculated above. Otherwise, leave it unchanged.
        self.args['warmup_steps'] = warmup_steps if self.args['warmup_steps'] == 0 else self.args['warmup_steps']
        
        # Return the total number of steps needed for training.
        return t_total

    def get_optimizer(self, model):
        """
        Returns the optimizer for training the given model.
    
        Args:
            model: The PyTorch model to be trained.
    
        Returns:
            An instance of the AdamW optimizer with the specified hyperparameters.
        """
    
        # Define the parameters that should not be decayed during optimization
        no_decay = ['bias', 'LayerNorm.weight']
    
        # Group the model parameters based on whether they should be decayed or not
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
        # Create an instance of the AdamW optimizer with the specified hyperparameters
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
    
        return optimizer
    

    
    def train(self, train_dataset):
        """Train the model on the given dataset"""
         # Set up logging
        writer = SummaryWriter()


        # Create the DataLoader for our training set
        train_dataloader = DataLoader(train_dataset, batch_size=self.args['train_batch_size'], shuffle=True)

        # Create the optimizer and the learning rate scheduler.
        optimizer = self.get_optimizer(self.model)
        total_steps = self.compute_training_time(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.args['warmup_steps'],
                                                    num_training_steps=total_steps)

        # Train the model
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(int(self.args['num_train_epochs']), desc="Epoch")
        print('Beginning training...')
        for epoch in train_iterator:
            epoch_loss = 0.0
            epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
            self.model.train()
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.model.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3]}
                outputs = self.model(**inputs)
                loss = outputs[0]
                if self.args['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args['gradient_accumulation_steps']
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
                # Accumulate the loss
                tr_loss += loss.item()
                # Update the model parameters every "gradient_accumulation_steps" batches
                if (step + 1) % self.args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
            
                    # Log the results every "logging_steps" steps
                    if self.args['logging_steps']  > 0 and global_step % self.args['logging_steps']  == 0:
                        writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        writer.add_scalar('loss', (tr_loss - logging_loss) / self.args['logging_steps'] , global_step)
                        logging_loss = tr_loss
            
                    # Save the model every "save_steps" steps
                    if self.args['save_steps']  > 0 and global_step % self.args['save_steps']  == 0:
                        output_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
                        os.makedirs(output_dir, exist_ok=True)
                        self.model.save_pretrained(output_dir)
                        torch.save(tr_loss, os.path.join(output_dir, 'training_loss.pt'))
                        torch.save(global_step, os.path.join(output_dir, 'global_step.pt'))

            epoch_loss = tr_loss / global_step
            writer.add_scalar('epoch_loss', epoch_loss, epoch)
            train_iterator.set_postfix(loss=epoch_loss)

        return global_step, tr_loss, self.model




# Instantiate the ModelPropaganda class with arguments
num_classes = 2
args = {
    'model_type': 'bert',
    'model_name': 'bert-base-uncased',
    'device': 'cpu'
}
model_propaganda = ModelPropaganda(num_classes, args)

# Print the loaded model
print(model_propaganda.model)