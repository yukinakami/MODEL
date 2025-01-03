import torch
import torch.nn as nn
import os
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import ViTConfig, ViTFeatureExtractor, ViTForImageClassification

from model.cnn import CNNEncoder

import numpy as np

class Model(nn.Module):
    def __init__(self, 
                 text_encoder = None,
                 image_encoder = None,
                 tokenizer = None,
                 config = None,
                 init_deit = True,):
        super().__init__()