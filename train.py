import torch
from model.init import weights_init
from dataset.dataset import dataset
from torch.utils.data import DataLoader
from model.net import classification
import torch.nn as nn
import cv2
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from model.net import model_resnet18
from torchvision.models import resnet18
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train_one_epoch(model,traindata,valdata,optimizer,loss,epoch,trainsize,valsize,epoch_end=10):
    model.train()
    print('Start Train')
    num = 0
    train_loss = 0
    with tqdm(total=len(traindata), desc=f'Epoch {epoch}/{epoch_end}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(traindata):
            images, targets = batch[0], batch[1]
            if torch.cuda.is_available():
                images = images.to('cuda')
                targets = targets.to('cuda')
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model(images).view(-1,2)
            predict = torch.argmax(outputs,axis = 1)
            num+=torch.sum(predict==targets).item()
            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value= loss(outputs, targets)
            train_loss += loss_value.item()
            pbar.set_postfix(**{'loss'      : loss_value.item(),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            # ----------------------#
            #   反向传播
            # ----------------------#
            loss_value.backward()
            optimizer.step()
    print('Finish Train')
    acc = num/trainsize
    train_acc = round(acc,4)
    print('train_acc:',acc)
    num = 0
    val_loss = 0
    model.eval()
    print('Start Validation')
    with tqdm(total=len(valdata), desc=f'Epoch {epoch}/{epoch_end}', postfix=dict, mininterval=0.3) as pbar:
        with torch.no_grad():

            num = 0
            for iteration, batch in enumerate(valdata):
                images, targets = batch[0], batch[1]
                if torch.cuda.is_available():
                    images = images.to('cuda')
                    targets = targets.to('cuda')
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model(images)
                outputs = model(images).view(-1, 2)
                predict = torch.argmax(outputs, axis=1)
                num += torch.sum(predict == targets).item()
                # ----------------------#
                #   计算损失
                # ----------------------#
                loss_value = loss(outputs, targets)
                val_loss += loss_value.item()
                pbar.set_postfix(**{'loss': loss_value.item(),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
            print('Finish Validation')
            acc = num / valsize
            val_acc = round(acc, 4)
            print('val_acc:', acc)
    torch.save(net, 'cpkt/res/ep%03d-train_acc%.3f-val_acc%.3f.pth' % (epoch , train_acc , val_acc))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device:',device)
    num_workers = 4
    bs = 4
    epoch = 10
    lr = 1e-4
    loss = nn.CrossEntropyLoss()
    # net = classification().to(device)
    # weights_init(net)
    net = model_resnet18(2).cuda()
    print(net)

    for param in net.parameters():
        param.requires_grad = False

    for param in net.fc.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(net.parameters(),lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    traindata = dataset('train','cpu')#,transform=transforms.Compose([
    #                    transforms.RandomHorizontalFlip(),
    #                    transforms.RandomVerticalFlip(),
    #                    transforms.RandomRotation(15),
    #                    transforms.RandomRotation(degrees=(-90, 90)),
    #                    transforms.Resize([150, 150]),
    #                    transforms.RandomCrop([120, 120]),] ))
    valdata = dataset('val','cpu')
    trainloader = DataLoader(traindata,batch_size=bs,shuffle=True,num_workers=num_workers)
    valloader = DataLoader(valdata, batch_size=bs, shuffle=True, num_workers=num_workers)
    for i in range(epoch):
        train_one_epoch(net,trainloader,valloader,optimizer,loss,i+1,len(traindata),len(valdata))
        lr_scheduler.step()


    for param in net.parameters():
        param.requires_grad = True
    net.fc.parameters()
    lr = 1e-5
    optimizer = optim.Adam(net.parameters(),lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    traindata = dataset('train','cpu')#,transform=transforms.Compose([
                       # transforms.RandomHorizontalFlip(),
                       # transforms.RandomVerticalFlip(),
                       # transforms.RandomRotation(15),
                       # transforms.RandomRotation(degrees=(-90, 90)),
                       # transforms.Resize([150, 150]),
                       # transforms.RandomCrop([120, 120]),] ))
    valdata = dataset('val','cpu')
    trainloader = DataLoader(traindata,batch_size=bs,shuffle=True,num_workers=num_workers)
    valloader = DataLoader(valdata, batch_size=bs, shuffle=True, num_workers=num_workers)
    for i in range(10,20):
        train_one_epoch(net,trainloader,valloader,optimizer,loss,i+1,len(traindata),len(valdata))
        lr_scheduler.step()

