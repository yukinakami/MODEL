# 新闻冲击的多模态钢铁价格风险预测模型
## 编码器设置
图像编码：cnn；文本编码：bert；时序数据编码：lstm  
图像特征降维：lstm  
文本、图像、维度统一为(x,768)
## 模态融合
3层cross-attention融合层
## 时序模型
4层lstm
## 预训练
损失函数：total_loss = 0.5 * mse + 0.3 * mae + 0.2 * huber
epoch = 10
学习率 = 1e-3
标签：标准化后的钢铁价格指数
## 数据集
使用clean2.json数据集，包含：  
1.有新闻的日期 -> 文本+至少一张图像+当天的价格指数  
2.没有新闻点日期 -> ‘当天无新闻’+纯黑图像+当天的价格指数  
共10662条数据，其中，2948条为无新闻数据，7714为有新闻数据，2154为唯一text(真实新闻数量)数量  
## 损失
2025-01-23 00:43:24,524 - 训练轮次||Epoch 1/10 - 损失||Training Loss: 0.7671  
2025-01-23 00:45:19,997 - 验证轮次||Epoch 1/10 - 损失||Validation Loss: 0.7430  
2025-01-23 02:00:46,452 - 训练轮次||Epoch 10/10 - 损失||Training Loss: 0.5749  
2025-01-23 02:02:44,399 - 验证轮次||Epoch 10/10 - 损失||Validation Loss: 0.7429  




