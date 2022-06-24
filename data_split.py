import random
import os
trainval = 0.9
train = 0.9

filename = ['./dataset/train.txt','./dataset/val.txt', './dataset/test.txt']
# datafile = './dataset/train'
# classes = ['calling','normal','smoking']
mask = 'dataset/self-built-masked-face-recognition-dataset/yes'
nomask = 'dataset/self-built-masked-face-recognition-dataset/no'

def split_train(file,classdir,classname):
    random.shuffle(classdir)
    size = len(classdir)
    # print(size)
    for i in range(0,int(size*trainval*train)):
        print(classname,':',i)
        file[0].write(classdir[i]+'\t'+classname+'\n')
    for i in range(int(size*trainval*train),int(size*trainval)):
        file[1].write(classdir[i]+'\t'+classname+'\n')
    for i in range(int(size * trainval),size):
        file[2].write(classdir[i]+'\t'+classname+'\n')

if __name__ == "__main__":
    random.seed(0)
    maskimgs = []
    nomasks = []
    imgdirs = os.listdir(mask)
    for i in imgdirs:
        pimgs = os.listdir(mask+'/'+i)
        for j in pimgs:
            maskimgs.append(mask+'/'+i+'/'+j)
    imgdirs = os.listdir(nomask)
    for i in imgdirs:
        pimgs = os.listdir(nomask+'/'+i)
        for j in pimgs:
            nomasks.append(nomask+'/'+i+'/'+j)
    file = []
    for i in filename:
        file.append(open(i, 'w'))
    split_train(file,maskimgs,'mask')
    split_train(file, nomasks, 'nomask')
    for i in file:
        i.close()
