import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import numpy as np
import pickle as p
import random
import matplotlib.pyplot as plt


num_epochs = 200

batch_size = 32

learning_rate = 1e-3

download = True ## True if you are running the code for the first time 

de_num = 1

cond_ch = 0

data_creation = True

repeat = 10


if not os.path.exists('./logs/images/samples'):
    os.makedirs('./logs/images/samples')
    
if data_creation == True:
    
    #### downloading and transforming the data #####
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    train_dataset = MNIST('./data',download=download, transform=img_transform)
    test_dataset = MNIST('./data',download=download,train=False, transform=img_transform)
    
    #### creating the training dataset #####
    dataset = [[] for i in range(10)]

    for i in range(len(dataset)):
        for data in train_dataset:
            if data[1]==i:
                dataset[i].append(data[0].numpy())

    class_sample_number = []
    for i in range(len(dataset)):
        class_sample_number.append(len(dataset[i]))


    train =  [[] for i in range(len(dataset))]

    for i in range(len(dataset)):
        train[i]= dataset[i][:min(class_sample_number)]

    train = np.array(train)
    
    #### creating the test dataset ####
    dataset = [[] for i in range(10)]

    for i in range(len(dataset)):
        for data in test_dataset:
            if data[1]==i:
                dataset[i].append(data[0].numpy())

    class_sample_number = []
    for i in range(len(dataset)):
        class_sample_number.append(len(dataset[i]))


    test =  [[] for i in range(len(dataset))]

    for i in range(len(dataset)):
        test[i]= dataset[i][:min(class_sample_number)]

    test = np.array(test)
    
    ### dumping the two the training and the testing dataset ######

    p.dump(test,open('MNIST_test','wb'))
    p.dump(train,open('MNIST_train','wb'))
    
else:
    MNIST_train =p.load(open('data/MNIST_train','rb'))
    MNIST_test  =p.load(open('data/MNIST_test','rb'))
    

DS_train_shape = MNIST_train.shape
MNIST_train = MNIST_train.reshape((DS_train_shape[0]*DS_train_shape[1],DS_train_shape[2],DS_train_shape[3],DS_train_shape[4]))

DS_train_shape = MNIST_test.shape
MNIST_test = MNIST_test.reshape((DS_train_shape[0]*DS_train_shape[1],DS_train_shape[2],DS_train_shape[3],DS_train_shape[4]))
    
print('MNIST training dataset size:  ',MNIST_train.shape)
print('MNIST testing dataset size:   ',MNIST_test.shape)


class encoder(nn.Module):

    def __init__(self,de_num,rate,cond_ch,complexity):

        super(encoder, self).__init__()
        
        self.de_num=de_num
        self.complexity=complexity
        self.rate = rate
        self.cond_ch= cond_ch

        self.encoder = nn.Sequential(
            nn.Conv2d(self.de_num, 16*self.complexity*self.rate, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16*self.complexity*self.rate, (8-self.cond_ch)*self.rate+self.de_num*self.cond_ch, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        
    def forward(self, x):
        z = self.encoder(x)    
        return z
    
    

class decoder(nn.Module):

    def __init__(self,rate,cond_ch,complexity):

        super(decoder, self).__init__()
        self.rate = rate
        self.cond_ch = cond_ch
        self.complexity= complexity

        self.decoder =nn.Sequential(
            nn.ConvTranspose2d((8-self.cond_ch)*self.rate+self.cond_ch, 16*self.complexity, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16*self.complexity, 8*self.complexity, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8*complexity, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
            )
        
    def forward(self, z):
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
        en_ch = (8-self.cond_ch)*self.rate #encoder chanels
        general_z = z[:,:en_ch]
        output = []
        for i in range(self.de_num):
            output.append(self.decoders[i](torch.cat((general_z,z[:,en_ch+i*self.cond_ch:en_ch+(i+1)*self.cond_ch]),1)))
            
        return output     
        
        
train_loss_log = [[] for i in range(repeat)]
test_loss_log = [[] for i in range(repeat)]

iter_per_epoch = int(len(MNIST_train)/(batch_size*de_num))
test_iter_per_epoch = int(len(MNIST_test)/(batch_size*de_num))

for k in range(repeat):
    
    model = autoencoder(de_num=de_num,rate=1,cond_ch=cond_ch,complexity=1).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.params, 
                                 lr=learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        index = random.sample(range(len(MNIST_train)),len(MNIST_train))
        epoch_loss = 0
        for i in range(iter_per_epoch):
            imgs = MNIST_train[index[i*batch_size*model.de_num:(i+1)*batch_size*model.de_num]]
            imgs = imgs.reshape((batch_size,model.de_num,28,28))
            imgs = torch.from_numpy(imgs)
            imgs = Variable(imgs).cuda()
            # ===================forward=====================
            output = model(imgs)
            loss = 0
            for i in range(model.de_num):
                loss+= criterion(output[i],imgs[:,i:(i+1),:,:]) 
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()/model.de_num

        # ===================log========================
        train_loss_log[k].append(epoch_loss/iter_per_epoch) 
        print('epoch:{},  loss:{:.4f}'
              .format(epoch+1, train_loss_log[k][-1]))

        index = random.sample(range(len(MNIST_test)),len(MNIST_test))
        test_loss = 0          
        for i in range(test_iter_per_epoch):
            imgs = MNIST_test[index[i*batch_size*model.de_num:(i+1)*batch_size*model.de_num]]
            imgs = imgs.reshape((batch_size,model.de_num,28,28))
            imgs = torch.from_numpy(imgs)
            imgs = Variable(imgs).cuda()
            # ===================forward=====================
            output = model(imgs)
            loss=0
            for i in range(model.de_num):
                loss+= criterion(output[i],imgs[:,i:(i+1),:,:]) 
            test_loss+= loss.item()/model.de_num

        test_loss_log[k].append(test_loss/test_iter_per_epoch)          


        if epoch%100==0:
            print('test loss:{:.4f}'.format(model.de_num, test_loss_log[k][-1]))
            ground_truth = imgs.cpu().detach().numpy()
            restored_imgs = []
            for i in range(len(output)):
                restored_imgs.append(output[i].cpu().detach().numpy())
            fig = plt.figure(figsize=(20,7))
            inx=1
            for i in range(model.de_num):
                ax = fig.add_subplot(5,4,inx)
                x = ground_truth[0,i:(i+1)][0,:,:]
                x = 0.5 * (x + 1)
                x = np.clip(x,0,1)
                ax.imshow(x)
                inx+=1
                ax = fig.add_subplot(5,4,inx)
                x = restored_imgs[i][0][0,:,:]
                x = 0.5 * (x + 1)
                x = np.clip(x,0,1)
                ax.imshow(x)
                inx+=1
            plt.savefig('./logs/images/samples/cifar10_baseline_imgs'+str(k)+'_'+str(epoch))

    
p.dump(test_loss_log,open('./logs/images/cifar10_baseline_test_loss_log','wb'))
p.dump(train_loss_log,open('./logs/images/cifar10_baseline_train_loss_log','wb'))