import torch.nn as nn
from torchsummary import summary
import torch
def make_lyaer(input,out,k=1,p=0,s=1):
    return nn.Sequential(
            nn.Conv2d(input, out, kernel_size=k, padding=p, stride=s, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(),)
class classification(nn.Module):
    def __init__(self):
        super(classification,self).__init__()
        self.layer1 = make_lyaer(3, 8, 3, 1, 2)
        # self.conv1 = nn.Conv2d(3,8,kernel_size=3,padding=1,stride=3,bias=False)
        # self.bn1 = nn.BatchNorm2d(8)
        # self.relu1 = nn.ReLU()
        self.layer2 = make_lyaer(8, 32,3,1,2)
        self.layer3 = make_lyaer(32, 128,3,1,2)
        self.layer4 = make_lyaer(128, 128)
        self.layer5 = make_lyaer(128, 3)
        self.conv6 = nn.Conv2d(3, 3, kernel_size=1, padding=0, stride=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.t1 = nn.Sigmoid()

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.t1(x)
        return x

def model_resnet18(num_class=3):
    from torchvision.models import resnet18
    from model.init import weights_init
    model = resnet18(pretrained=False)
    model.load_state_dict(torch.load('resnet18.pth'))
    model.fc = weights_init(nn.Linear(512, num_class, bias=False),'xavier')
    return model
if __name__ == "__main__":
    import torch
    import cv2
    import numpy as np
    # net = classification()
    # net.to('cuda')
    # model_dict      = net.state_dict()
    # pretrained_dict = torch.load('../cpkt/ep020-loss0.762-val_loss0.778.pth', map_location = 'cuda')
    # net.load_state_dict(pretrained_dict.state_dict())
    # img = cv2.imread('../dataset/train/smoking_images/18.jpg')
    # shape = img.shape
    # img = torch.tensor(np.transpose((cv2.imread('../dataset/train/smoking_images/18.jpg') / 255.0), (2, 0, 1)),
    #              device='cuda', dtype=torch.float32)
    # img = img.view(1,3,shape[0],shape[1])
    # x,y = net(img)
    # print(x)
    # print(y.shape)
    # from torchvision.models import resnet2
    # model = resnet2(pretrained=True)
    # from torchsummary import summary
    # # summary(model,(3,224,224))
    # # model = nn.Sequential(*list(model.modules())[:-2])
    model = model_resnet18(3)
    # summary(model, (3, 120, 120),batch_size=32,device='cpu')
    print(model)
    # print(model_resnet18())
