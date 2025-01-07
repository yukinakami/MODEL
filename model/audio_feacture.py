import torch
import torch.nn as nn
import torchaudio
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, seq_len, dropout=0.1, sample_rate=16000, n_fft=400, hop_length=160, n_mels=23):
        super(AudioEncoder, self).__init__()

        # 输入特征维度，隐藏层维度，注意力头数量，编码器层数，序列长度，丢弃率
        self.input_dim = input_dim  # 输入特征维度
        self.hidden_dim = hidden_dim  # 隐藏层维度,规定为768
        self.num_layers = num_layers  # 编码器层数
        self.num_heads = num_heads  # 注意力头数量
        self.seq_len = seq_len  # 序列长度
        self.dropout = dropout

        # 设置STFT相关参数
        self.sample_rate = sample_rate  # 音频采样率
        self.n_fft = n_fft  # STFT窗口大小
        self.hop_length = hop_length  # 跳步长度
        self.n_mels = n_mels  # 梅尔频带数量

        # 定义一个STFT层（音频特征提取器）=> STFT + Mel Spectrogram
        self.stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        
        # 添加一个线性层来投影特征到Transformer的输入维度
        self.feature_projection = nn.Linear(n_mels, hidden_dim)

        # 定义一个Transformer编码器层
        self.encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,  # 设置为hidden_dim
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,  # 前馈神经网络的隐藏层维度
            dropout=dropout
        )
        
        # 定义一个Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        
        # 定义一个全连接层作为输出层，用于处理Transformer编码器的输出
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)  # 改为hidden_dim -> hidden_dim

    def forward(self, x):
        # x为输入的音频数据，形状为(batch_size, num_samples)
        # 使用STFT转换为梅尔频谱图，输出形状为(batch_size, n_mels, num_frames)
        
        # 使用STFT层提取音频特征
        mel_spectrogram = self.stft(x)
        
        # 转置为 (batch_size, num_frames, n_mels)，因为我们需要 seq_len 和 hidden_dim
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)  # (batch_size, num_frames, n_mels)

        # 将梅尔频谱图通过线性层映射到hidden_dim
        x = self.feature_projection(mel_spectrogram)  # 维度变换为 (batch_size, num_frames, hidden_dim)
        
        # 将输入转换为 Transformer 需要的形状: (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)  # (num_frames, batch_size, hidden_dim)

        # 通过Transformer编码器
        x = self.transformer_encoder(x)

        # Flatten为 (batch_size * seq_len, hidden_dim)，以便输入到输出层
        x = x.view(-1, self.hidden_dim)  # (batch_size * seq_len, hidden_dim),将输出展平

        # 通过全连接层处理Transformer的输出
        x = self.output_layer(x)

        # 恢复到 (seq_len, batch_size, hidden_dim) 的形状，如果需要返回这样的形状
        #x = x.view(self.seq_len, -1, self.hidden_dim)  # (seq_len, batch_size, hidden_dim)

        return x

if __name__ == '__main__':
    # 测试AudioEncoder
    audio_encoder = AudioEncoder(input_dim=13, hidden_dim=768, num_heads=8, num_layers=6, seq_len=100, sample_rate=16000, n_mels=23)
    audio_data = torch.randn(32, 16000)  # 假设输入batch_size=32，音频长度=16000
    audio_features = audio_encoder(audio_data)
    print(audio_features.shape)  # 应输出 torch.Size([100, 32, 768])（假设seq_len=100）
