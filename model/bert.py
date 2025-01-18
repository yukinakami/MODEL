from transformers import BertTokenizer, BertModel
import torch

class BertEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask, max_length=512):
        # Pass tokens through BERT model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Return the CLS token representation
        return outputs.last_hidden_state[:, 0, :]

# if __name__ == "__main__":
#     encoder = BertEncoder()
#     text = "This is a test sentence."
#     representation = encoder.encode(text)
#     print(representation)  
