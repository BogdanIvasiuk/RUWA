import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PredictPropaganda:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict_propaganda(self, logits):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)

        # Get the index of the highest probability
        pred_class = probs.argmax().item()

        # Check if the highest probability belongs to the positive class (index 1 in this case)
        if pred_class == 1:
            return True
        else:
            return False

    def check_sentence_propaganda(self, sentence):
        input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)])
        attention_mask = torch.tensor([[1] * len(input_ids[0])])
        token_type_ids = torch.zeros_like(input_ids)

        # Send the inputs to the model and get the predictions
        with torch.no_grad():
            inputs = {'input_ids': input_ids.to(self.device ),
                      'attention_mask': attention_mask.to(self.device ),
                      'token_type_ids': token_type_ids.to(self.device )}
            outputs = self.model(**inputs)
            logits = outputs[0]

        return self.predict_propaganda(logits)

    def propaganda_detection(self, sentences):
        propaganda_indices = []
        numbered_body = ""
        for i, sentence in enumerate(sentences):
            numbered_body += f"{i+1}) {sentence.strip()}\n"
            try:
                if self.check_sentence_propaganda(sentence):
                    propaganda_indices.append(i+1)
            except Exception as e:
                print(f"Error in propaganda detection with sentence {i}: {sentence}")
                print(f"Error message: {e}")

        if propaganda_indices:
            propaganda_sentence_numbers = ', '.join(str(i) for i in propaganda_indices)
            propaganda_text = f"The sentences number: {propaganda_sentence_numbers} contain propaganda"
        else:
            propaganda_text = "No propaganda found"
        

        return propaganda_text, numbered_body, propaganda_indices





class CalculateSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('stsb-mpnet-base-v2')
    
    def get_embeddings(self, sentences):
        return self.model.encode(sentences, show_progress_bar=True)

    
    def get_similarity(self, s1, s2):
        # Calcola gli embedding delle frasi
        sentence_embs = self.model.encode([s1, s2], show_progress_bar=True)
        
        # Calcola la matrice di similarità
        similarity_matrix = cosine_similarity(sentence_embs)
        
        # Restituisce la similarità tra le due frasi
        return similarity_matrix[0][1]
    
    def get_mean_similarity(self, sentence_embs1, sentence_embs2, min_num_sentences=0):
        
        # Calcola la matrice di similarità
        similarity_matrix = cosine_similarity(sentence_embs1,sentence_embs2)

        
        # Calcola la media delle similitudini
        if min_num_sentences != 0:
            similarity_scores = similarity_matrix.flatten()
            sorted_scores = sorted(similarity_scores, reverse=True)
            mean_score = sum(sorted_scores[:min_num_sentences]) / min_num_sentences
        else:
            mean_score = 0
        
        # Restituisce la media delle similitudini
        return mean_score


