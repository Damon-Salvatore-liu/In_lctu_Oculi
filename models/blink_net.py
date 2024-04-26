import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import yaml
import os
from easydict import EasyDict as edict

class BlinkCNN(nn.Module):
    """
    CNN for eye blinking detection
    """
    def __init__(self):
        super(BlinkCNN, self).__init__()
        # 加载预训练的VGG16模型
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg16.classifier[6] = nn.Linear(4096, 2) # 分类数设为2

    def forward(self, x):
        pred = self.vgg16(x)
        return pred
    
    def _initialize():
        # 对模型参数进行初始化
        pass
    
    
    
class BlinkLRCN(nn.Module):
    """
    LRCN for eye blinking detection
    """
    def __init__(self, cfg_path):
        super(BlinkLRCN, self).__init__()
        with open(cfg_path, 'r') as f:
            cfg = edict(yaml.load(f, Loader=yaml.Loader))
        self.cfg = cfg
        self.num_classes = cfg.NUM_CLASS
        self.max_time = cfg.MAX_TIME
        self.hidden_unit = cfg.HIDDEN_UNIT
        # 定义模型层
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg16.classifier = self.vgg16.classifier[:4] # 取特征层
        self.rnn = nn.LSTM(input_size=4096, hidden_size=cfg.HIDDEN_UNIT, batch_first=True)
        self.fc = nn.Linear(256, cfg.NUM_CLASS)

        self.layers = {}
        self.params = {}
    
    def forward(self, x):
        # x.shape 为 BxTxCxHxW to (BxT)xCxHxW
        C, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.view(-1, C, H, W)
        x = self.vgg16(x)
        cnn_out = x.view(-1, self.max_time, 4096)
        
        rnn_out, _ = self.rnn(cnn_out) # shape: BxTx256
        rnn_out = rnn_out.reshape(-1, self.hidden_unit) # shape: (BxT)x256
        
        out = self.fc(rnn_out) # (BxT)x2
        out = out.view(-1, self.max_time, self.num_classes)
        return out
        
        
        
        
        
    
    def _initialize():
        # 对模型参数进行初始化
        pass
    
    

if __name__ == "__main__":
    # net=vgg16(weights=VGG16_Weights.DEFAULT)
    # net.classifier = net.classifier[:4]
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = BlinkLRCN('../configs/blink_lrcn.yml').to(device)
    input = torch.randn(4, 20, 3, 224, 224).to(device)
    output = net(input)
    print(output)
    print(output.shape)
    # print(net)
    y = torch.randint(0, 2, (4, 20)).to(device)
    y = y.unsqueeze(2)
    loss = loss_fn(output, y)
    print(loss)
    print(y.shape)