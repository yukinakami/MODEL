from transformers import BertTokenizer, BertModel
import torch

class BertEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)


    def encode(self, input_ids, attention_mask):
        # Pass tokens through BERT model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Return the CLS token representation
        return outputs.last_hidden_state[:, 0, :]

if __name__ == "__main__":
    encoder = BertEncoder()    
    # Example text
    text = "This is a test sentence."    
    # Tokenize the input text
    encoding = encoder.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Extract input_ids and attention_mask
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']   
    # Get the BERT representation
    representation = encoder.encode(input_ids, attention_mask)  
    # Print the representation
    print(representation.shape)
