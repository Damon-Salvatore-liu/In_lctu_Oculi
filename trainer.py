import os
import yaml
import cv2
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import SGD, Adam, lr_scheduler

class Trainer(object):
    def __init__(self, cfg_path, model, mode='cnn'):
        
        with open(cfg_path, 'r') as f:
            cfg = edict(yaml.load(f, Loader=yaml.Loader))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.mode = mode
        
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.opt = self._set_optimizer()
        
        
    def train(self, train_loader, test_loader, *args):
        if self.mode == 'cnn':
            return self._train_cnn(train_loader, test_loader)
        elif self.mode == 'lrcn':
            return self._train_lrcn(seq_tensor=args[0],
                                   len_list=args[1],
                                   state_list=args[2])
        else:
            raise ValueError('We only support mode = [cnn, lrcn]...')
        
    def test(self, *args):
        if self.mode == 'cnn':
            return self._test_cnn(images=args[0])
        elif self.mode == 'lrcn':
            return self._test_lrcn(seq_tensor=args[0],
                                   len_list=args[1])
        else:
            raise ValueError('We only support mode = [cnn, lrcn]...')
        
    def _train_cnn(self, train_loader, test_loader):
        epochs = self.cfg.TRAIN.NUM_EPOCH
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', dynamic_ncols=True) as tqdm_loader:
                for images, labels in tqdm_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    pred = self.model(images)
                    loss = self.loss(pred, labels)
                    loss.backward()
                    self.opt.step()
                    epoch_loss += loss.item()
                    tqdm_loader.set_postfix(ordered_dict={"step_loss": loss.item()})
            

            if (epoch + 1) % self.cfg.TRAIN.SAVE_INTERVAL == 0:
                self._save_model(epoch)
            acc = self.eval(test_loader)
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc}")
    
    def eval(self, test_loader):
        self.model.eval()
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.cfg.NUM_CLASS).to(self.device)
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.model(images)
                acc(pred, labels)
                
            _acc = acc.compute()
        return _acc
                
                
                
                
        
    
    def _train_lrcn(self):
        pass
    
    def _test_cnn(self, images):
        for i, im in enumerate(images):
            images[i] = cv2.resize(im, (self.img_size[0], self.img_size[1]))
            
            
    
    def _test_lrcn(self):
        pass
    
    def _set_optimizer(self):
        if self.cfg.TRAIN.METHOD == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=self.cfg.TRAIN.LEARNING_RATE)
        elif self.cfg.TRAIN.METHOD == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=self.cfg.TRAIN.LEARNING_RATE)
        else:
            raise ValueError('We only support [SGD, Adam] right now...')
        
        return optimizer
    
    def _save_model(self, epoch):
        save_path = os.path.join(self.cfg.TRAIN.SAVE_PATH, 'blink_'+ self.mode +'_'+str(epoch+1)+'.pt')
        torch.save(self.model.state_dict(), save_path)
        
if __name__ == '__main__':
    with open('./configs/blink_cnn.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    print('完成')