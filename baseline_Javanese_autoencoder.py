import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import numpy as np
import pickle 
import random
import matplotlib.pyplot as plt
import os 
import soundfile as sf
import librosa

num_epochs = 20

batch_size = 14

learning_rate = 1e-3

de_num = 1

cond_ch=0

complexity = 1

repeat = 6

fs = 18000

length = 4

stride = 4

metadata = np.array(pickle.load(open( "javanese_metadata", "rb" )))

train_ind = int(len(metadata)*0.85)

metatrain = metadata[:train_ind]
metatest  = metadata[train_ind:]


class TrainingDataset(Dataset):
    def __init__(self,metadata,fs,length,de_num):
        self.DS = metadata
        self.de_num = de_num
        self.fs = fs
        self.length = length
        self.len = int(len(self.DS)/de_num)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        files_name = [self.DS[idx]]
        for i in range(1,self.de_num):
            files_name.append(self.DS[i*self.len+idx])
        samples = []
        for i in range(self.de_num):
            samples.append(librosa.load(files_name[i], sr=self.fs)[0][:self.fs*self.length])
        samples = np.array(samples)
        mean = samples.mean(-1)
        std = samples.std(-1)
        data_norm = np.array((samples - mean[:,np.newaxis])/std[:,np.newaxis])
        return data_norm
    
training = TrainingDataset(metatrain,fs,length,de_num)
train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)

testing = TrainingDataset(metatest,fs,length,de_num)
test_loader = DataLoader(testing, batch_size=batch_size, shuffle=True)    


class encoder(nn.Module):
    
    def __init__(self,de_num, rate, cond_ch, complexity):
        super(encoder,self).__init__()
        
        self.de_num=de_num
        self.complexity=complexity
        self.rate = rate
        self.cond_ch= cond_ch
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1*self.de_num, 128*self.rate*self.complexity, 9, stride=4,dilation = 3 , padding=12),  
            nn.ReLU(True),
            nn.Conv1d(128*self.rate*self.complexity,512*self.rate*self.complexity, 9, stride=4,dilation = 3 , padding=12),  
            nn.ReLU(True),
            nn.Conv1d(512*self.rate*self.complexity,(16-self.cond_ch)*self.rate+self.de_num*self.cond_ch, 9, stride=4,dilation = 3 , padding=12)
        )
        
    def forward(self,x):
        z = self.encoder(x)
        
        return z
    
    
class decoder(nn.Module):
        
    def __init__(self, rate, cond_ch, complexity):
        super(decoder,self).__init__()
        
        self.rate = rate
        self.cond_ch = cond_ch
        self.complexity= complexity
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d((16-self.cond_ch)*self.rate+self.cond_ch,512*self.complexity, 10, stride=4,dilation = 3,padding=12),  
            nn.ReLU(True),
            nn.ConvTranspose1d(512*self.complexity, 128*self.complexity, 10, stride=4,dilation =3, padding=12),  
            nn.ReLU(True),
            nn.ConvTranspose1d(128*self.complexity, 1, 10, stride=4,dilation = 3 , padding=12)
        )
        
    def forward(self,z):
        y = self.decoder(z)
        return y
    
    
class autoencoder(nn.Module):

    def __init__(self,de_num,rate,cond_ch,complexity):

        super(autoencoder, self).__init__()
        self.rate=rate
        self.de_num=de_num
        self.complexity=complexity
        self.cond_ch=cond_ch

        self.encoder = encoder(self.de_num,self.rate,self.cond_ch,self.complexity)
        self.params=[p for p in self.encoder.parameters()]
        self.decoders=[]
        for i in range(de_num):
            d = decoder(self.rate,self.cond_ch,self.complexity).cuda()
            self.decoders.append(d)
            for p in d.parameters():   
                self.params.append(p)

    def forward(self, x):
        
        z = self.encoder(x)
        en_ch = (16-self.cond_ch)*self.rate #encoder chanels
        general_z = z[:,:en_ch]
        output = []
        for i in range(self.de_num):
            output.append(self.decoders[i](torch.cat((general_z,z[:,en_ch+i*self.cond_ch:en_ch+(i+1)*self.cond_ch]),1)))
            
        return output     



train_loss_log = [[] for i in range(repeat)]
test_loss_log = [[] for i in range(repeat)]

for k in range(repeat):
    
    model = autoencoder(de_num,de_num,cond_ch,complexity).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.params, 
                                 lr=learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            auds = data.float()
            auds = Variable(auds).cuda()
            # ===================forward=====================
            output = model(auds)
            loss = 0
            for i in range(model.de_num):
                loss+= criterion(output[i],auds[:,i:i+1,:]) 
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()/model.de_num

        # ===================log========================
        train_loss_log[k].append(epoch_loss/len(train_loader)) 
        print('Repeat:{},  epoch:{},  loss:{:.4f}'
              .format(k, epoch+1, train_loss_log[k][-1]))


        test_loss = 0          
        for data in test_loader:
            auds = data.float()
            auds = Variable(auds).cuda()
            # ===================forward=====================
            output = model(auds)
            loss = 0
            for i in range(model.de_num):
                loss+= criterion(output[i],auds[:,i:i+1,:]) 
            test_loss+= loss.item()/model.de_num

        test_loss_log[k].append(test_loss/len(test_loader))          


        if epoch%4==0:
            print('Repeat:{},  test loss:{:.4f}'.format(k, test_loss_log[k][-1]))
            ground_truth = auds.cpu().data
            restored_aud = output[0].cpu().data
            sf.write('./logs/audio/records/javanese_baseline_ground_truth_audio'+str(k)+'_'+str(epoch)+'.wav', ground_truth[0,0], fs)
            sf.write('./logs/audio/records/javanese_baseline_predicted_audio'+str(k)+'_'+str(epoch)+'.wav', restored_aud[0,0], fs)

    pickle.dump(test_loss_log,open('logs/audio/javanese_baseline_test_loss_log','wb'))
    pickle.dump(train_loss_log,open('logs/audio/javanese_baseline_train_loss_log','wb'))