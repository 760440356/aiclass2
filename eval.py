import torch
from model.init import weights_init
from dataset.dataset import dataset
from torch.utils.data import DataLoader
from model.net import classification
import torch.nn as nn
import cv2
import torch.optim as optim
from tqdm import tqdm
from model.net import model_resnet18
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
def eval(model,valdata,valsize):
    num = 0
    val_loss = 0

    model.eval()
    print('Start Test')
    with tqdm(total=len(valdata), postfix=dict, mininterval=0.3) as pbar:
        with torch.no_grad():
            masknum = 0
            nomasknum = 0
            rmasknum = 0
            rnomasknum = 0
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
                masknum += torch.sum(targets == 0).item()
                nomasknum += torch.sum(targets == 1).item()
                rmasknum += torch.sum((predict == targets).mul(targets == 0)).item()
                rnomasknum += torch.sum((predict == targets).mul(targets == 1)).item()
                # ----------------------#
                #   计算损失
                # ----------------------#
                pbar.update(1)
            print('Finish Test')
            acc = num / valsize
            print('val_acc:', acc)
            print('mask_acc:', rmasknum/masknum)
            print('nomask_acc:', rnomasknum / nomasknum)
            f.write('val_acc:{}\n'.format(acc))
            f.write('mask_num:{},rmask_num{},mask_acc{}\n'.format(masknum,rmasknum,masknum/rmasknum))
            f.write('nomask_num:{},rnomask_num{},nomask_acc{}\n'.format(nomasknum, rnomasknum, nomasknum / rnomasknum))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device:',device)
    num_workers = 4
    bs = 4
    # epoch = 20
    # lr = 1e-3
    # loss = nn.CrossEntropyLoss()
    # net = classification().to(device)
    # net = torch.load('cpkt/ep020-loss0.762-val_loss0.778.pth')
    # net.to(device)
    # net = model_resnet18().cuda()
    # net.load()
    import os
    pthdir = os.listdir('cpkt/res')
    f = open('res.txt', 'w')
    for i in pthdir:
        print(i)
        f.write('cpkt/res/'+i+'\n')
        net = torch.load('cpkt/res/'+i).cuda()
        testdata = dataset('test','cpu')
        testloader = DataLoader(testdata, batch_size=bs, shuffle=True, num_workers=num_workers)
        eval(net, testloader, len(testdata))
    f.close()
