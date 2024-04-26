import sys
sys.path.append('..')

from models.blink_net import BlinkLRCN
from trainer import Trainer
from proc_data.seq_data import SeqDataset # 加载不了

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BlinkLRCN('../configs/blink_lrcn.yml').to(device)
    
    
    trans = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                ])
    trainer = Trainer('../configs/blink_lrcn.yml', model, 'lrcn')
    train_data = SeqDataset(anno_path='../datas/cnn/train.p', 
                            data_dir='../datas/cnn/',
                            transforms=trans
                            )
    test_data = SeqDataset(anno_path='../datas/cnn/test.p', 
                            data_dir='../datas/cnn/',
                            transforms=trans)
    
    train_dataloader = DataLoader(train_data, batch_size=32,shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    print('Training')
    trainer.train(train_dataloader, test_loader)
            
            
        
    
    


if __name__ == '__main__':
    main()