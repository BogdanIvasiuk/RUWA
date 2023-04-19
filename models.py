from info import MODEL_CLASSES
import torch

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
    
    def train(self, train_dataset):
        """Train the model on the given dataset"""
        # Set the seed for reproducibility
        torch.manual_seed(self.args['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args['seed'])

        # Create the DataLoader for our training set
        train_dataloader = DataLoader(train_dataset, batch_size=self.args['train_batch_size'], shuffle=True)

        # Create the optimizer and the learning rate scheduler.
        optimizer = AdamW(self.model.parameters(), lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
        total_steps = len(train_dataloader) * self.args['num_train_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.args['warmup_steps'],
                                                    num_training_steps=total_steps)

        # Train the model
        print('Beginning training...')
        for epoch in range(1, self.args['num_train_epochs'] + 1):
            epoch_loss = 0.0
            self.model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                input_ids, attention_mask, labels = tuple(t.to(self.model.device) for t in batch)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                epoch_loss += loss.item()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f'Epoch {epoch} loss: {avg_epoch_loss:.4f}')

        print('Training complete!')




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