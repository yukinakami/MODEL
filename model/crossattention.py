import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):

        super(CrossAttentionFusion, self).__init__()
        self.query_text = nn.Linear(input_dim, hidden_dim)
        self.key_image = nn.Linear(input_dim, hidden_dim)
        self.value_image = nn.Linear(input_dim, hidden_dim)

        self.query_image = nn.Linear(input_dim, hidden_dim)
        self.key_text = nn.Linear(input_dim, hidden_dim)
        self.value_text = nn.Linear(input_dim, hidden_dim)

        self.output_fc = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, text_features, image_features):

        query_text = self.query_text(text_features)  # (batch_size, seq_len_text, hidden_dim)
        key_image = self.key_image(image_features)  # (batch_size, seq_len_image, hidden_dim)
        value_image = self.value_image(image_features)  # (batch_size, seq_len_image, hidden_dim)

        attention_weights_text_to_image = F.softmax(
            torch.matmul(query_text, key_image.transpose(-2, -1)), dim=-1
        )  # (batch_size, seq_len_text, seq_len_image)
        attended_image = torch.matmul(attention_weights_text_to_image, value_image)  # (batch_size, seq_len_text, hidden_dim)

        
        query_image = self.query_image(image_features)  # (batch_size, seq_len_image, hidden_dim)
        key_text = self.key_text(text_features)  # (batch_size, seq_len_text, hidden_dim)
        value_text = self.value_text(text_features)  # (batch_size, seq_len_text, hidden_dim)

        attention_weights_image_to_text = F.softmax(
            torch.matmul(query_image, key_text.transpose(-2, -1)), dim=-1
        )  # (batch_size, seq_len_image, seq_len_text)
        attended_text = torch.matmul(attention_weights_image_to_text, value_text)  # (batch_size, seq_len_image, hidden_dim)

        
        fused_features = torch.cat([attended_image.mean(dim=1), attended_text.mean(dim=1)], dim=-1)  # (batch_size, 2 * hidden_dim)

        
        output_features = self.output_fc(fused_features)  # (batch_size, input_dim)

        return output_features
