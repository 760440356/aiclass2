import torch
import cv2
import numpy as np
import torch.nn as nn
datalist = ['calling','normal','smoking']
if __name__ =="__main__":
    net = torch.load('cpkt/resnet1/ep020-train_acc0.998-val_acc0.771.pth').to('cpu')
    img_path = 'q.jpeg'
    img = cv2.imread(img_path)
    img = torch.tensor(np.transpose(img/255,(2,0,1)),device='cpu',dtype=torch.float32).view(-1,3,img.shape[0],img.shape[1])
    net.eval()
    result = net(img)
    softmax = nn.Softmax()
    # result = softmax(result)
    print(net)
    print(result)
    print(datalist[torch.argmax(result).item()])