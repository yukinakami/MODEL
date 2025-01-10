import torch
import torch.nn as nn #����������ģ��
import os
import torch.optim as optim #�����Ż���
from torch.utils.data import DataLoader, Dataset #�������ݼ�����/���ݼ�
from model.model import MultimodalModel #����ģ��

#�������ݼ�
class MultimodalDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            json = f.readlines()
        self.data = json.load(f)

    def __len__(self):
        return len(self.data) #�������ݳ���

    def __getitem__(self, idx):
        sample = self.data[idx] #��ȡ����
        return {
            'text': torch.tensor(sample['text'], dtype=torch.float32), #�ı�����
            'image': torch.tensor(sample['image'], dtype=torch.float32), #ͼ������
            'data': torch.tensor(sample['data'], dtype=torch.float32), #ʱ������
            'audio': torch.tensor(sample['audio'], dtype=torch.float32), #��Ƶ����
        }

#ѵ������
def train_model(model, dataloader, optimizer, device):
    #ģ�ͣ����ݼ���������ʧ�������Ż������豸
    model.train() #ѵ��ģʽ
    total_loss = 0.0 #��ʼ����ʧ

    criterion = nn.MSELoss() #������ʧ����

    for batch in dataloader:
        text = batch['text'].to(device)
        image = batch['image'].to(device)
        data = batch['data'].to(device)
        audio = batch['audio'].to(device)
        original = batch['original'].to(device) #ԭʼ����

        optimizer.zero_grad() #�ݶ�����

        outputs = model(text, image, data, audio) #ǰ�򴫲�,�ó����

        #MSE��ʧ��������
        loss = criterion(outputs, original) #������ʧ
        loss.backward() #���򴫲�
        optimizer.step() #���²���

        total_loss += loss.item() #�ۼ���ʧ

    return total_loss / len(dataloader) #����ƽ����ʧ

def valiate_model(model, dataloader, criterion, device):
    model.eval() #����ģʽ
    total_loss = 0.0 #��ʼ����ʧ

    with torch.no_grad(): #�������ݶ�
        for batch in dataloader:
            text = batch['text'].to(device)
            image = batch['image'].to(device)
            data = batch['data'].to(device)
            audio = batch['audio'].to(device)
            original = batch['original'].to(device) #ԭʼ����

            outputs = model(text, image, data, audio) #ǰ�򴫲�,�ó����

            loss = criterion(outputs, original) #������ʧ
            total_loss += loss.item() #�ۼ���ʧ
        
    return total_loss / len(dataloader) #����ƽ����ʧ

def main():
    #����ѵ���豸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #��������
    train_json = '#' #ѵ������·��
    dataset = MultimodalDataset(train_json) #���ݼ�

    #��̬����ѵ��������֤��
    train_size = int(0.8 * len(dataset)) #ѵ������СΪ���ݼ���80%
    val_size = len(dataset) - train_size #��֤����СΪ���ݼ���20%
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) #����������ݼ�

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    #ѵ�����ݼ�����, batch_size=32��˼��ÿ��ѵ��32������, shuffle=True��˼��ÿ��ѵ��ǰ��������
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False) #��֤���ݼ�����  

    #����ģ��
    model = MultimodalModel().to(device) #ģ��

    #�����Ż���
    optimizer = optim.Adam(model.parameters(), lr=0.001) #�Ż���, Adam�Ż���, ѧϰ��Ϊ0.001
    
    #ѵ������
    num_epochs = 10 #ѵ������
    checkpoint_path = '#' #ģ�ͱ���·��
    os.makedirs(checkpoint_path, exist_ok=True) #����ģ�ͱ���·��

    #ѵ��
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_dataloader, optimizer, device)
        val_loss = valiate_model(model, val_dataloader, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        #��ӡѵ����Ϣ��������ǰ��������������ѵ����ʧ����֤��ʧ������4λС��

        #����ģ��
        checkpoint_path = os.path.join(checkpoint_path, f'model_{epoch + 1}.pth')
        #ģ�ͱ���·����ģ������
        torch.save(model.state_dict(), checkpoint_path)

    print('Training Finished!') #ѵ������

if __name__ == '__main__':
    main() #������