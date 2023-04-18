from transformers import BertTokenizer, BertForSequenceClassification
import torch

class BertModel:
    def __init__(self, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        
    def train(self, train_dataset, epochs=3, batch_size=16):
        # Definiamo l'optimizer e il loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Creiamo il dataloader per il training set
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Iteriamo sull'intero dataset per un certo numero di epoche
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(train_dataloader):
                # Mettiamo i dati sulla GPU (se disponibile)
                batch = [r.to(device) for r in batch]
                
                input_ids, attention_mask, labels = batch
                
                # Reset dei gradienti
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Aggiornamento dei parametri del modello
                optimizer.step()
                
                if step % 100 == 0 and step != 0:
                    print(f"Step {step}: loss = {loss.item()}")
        
        # Salviamo il modello
        self.model.save_pretrained("./bert_model")
        self.tokenizer.save_pretrained("./bert_model")
        
    def load(self, path):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path)
        
    def predict(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt")
        outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
        return prediction.item()



model = BertModel(num_labels=2)
train_dataset = ""

model.train(train_dataset=train_dataset, epochs=3, batch_size=16)
model = BertModel(num_labels=2)
model.load(path="./bert_model")
sentence = "This is a test."
prediction = model.predict(sentence)
print(prediction)
