from transformers import BertTokenizer, BertModel
import torch

class BertEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, text, max_length=512):
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Pass tokens through BERT model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Return the CLS token representation
        return outputs.last_hidden_state[:, 0, :]

# if __name__ == "__main__":
#     encoder = BertEncoder()
#     text = "This is a test sentence."
#     representation = encoder.encode(text)
#     print(representation)  
