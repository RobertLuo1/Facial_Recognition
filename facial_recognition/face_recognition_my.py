from facenet_pytorch import MTCNN
import os
import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
import face_model as fm
import numpy as np
import matplotlib.pyplot as plt
from data_generator import MyDataset

model_File = r"./model"
feature_File = r"./feature.pth"
test_File = r"./test"
train_File = r"./train"
picture_File = r"./pictures"

mtcnn0 = MTCNN(image_size=240,min_face_size=40,keep_all=False)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True

def GetName():
    for dirpaths,dirnames,filenames in os.walk(train_File):
        return dirnames


Mydataset = MyDataset(train_File)
test_n_samples = len(Mydataset)/6
train_n_samples = len(Mydataset)-test_n_samples
names = GetName()
classes = len(names)
torch.save(names,model_File+"//"+'name.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_val_dataset(dataset):
    np.random.seed(42)
    rand_number = np.random.randint(0,5)
    data_index =list(range(len(dataset)))
    test_idx = data_index[rand_number:len(dataset):6]
    train_set = set(data_index)-set(test_idx)
    train_idx = list(train_set)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    print("the size of train set is "+str(len(datasets['train'])))
    datasets['test'] = Subset(dataset, test_idx)
    print("the size of test set is "+str(len(datasets['test'])))
    return datasets

Mydatasets = train_val_dataset(Mydataset)
# make_test_folder(Mydatasets['test'])
trainloader = DataLoader(Mydatasets['train'],shuffle=True,batch_size=4)
testloader = DataLoader(Mydatasets['test'])

def train():
    resnet = fm.InceptionResnetV1(classify=True, num_classes=classes, device=device, pre_trained=True).train()
    epochs = 10
    optim_para = []
    criterion = nn.CrossEntropyLoss()
    for para in resnet.parameters():
        para.requires_grad = True
    # for para in resnet.last_linear.parameters():
    #     para.requires_grad = True
    #     optim_para.append(para)
    # for para in resnet.last_bn.parameters():
    #     para.requires_grad = True
    #     optim_para.append(para)
    for para in resnet.logits.parameters():
        para.requires_grad = True
        optim_para.append(para)
    optimizer = optim.Adam(optim_para, lr=0.001)
    train_losses = []
    accuracy = []
    best_acc = 0
    best_epoch = 0
    for e in range(epochs):
        running_loss = 0.0
        for image, idx in trainloader:
            out = resnet(image.to(device))
            loss = criterion(out, idx.to(device))
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))
        acc = evaluate(resnet, testloader)
        accuracy.append(acc)
        if e % 5 ==0:
            print(f"The epoch is {e} "+f"The loss is{running_loss / len(trainloader)} "+f"The accuracy is {acc}")
        if acc > best_acc:
            torch.save(resnet.state_dict(),model_File+"//"+'resnet.pth')
            best_acc = acc
            best_epoch = e
    plt.title("Train_loss and accuracy")
    plt.plot(train_losses, label='Training loss', )
    plt.plot(accuracy, label="accuracy", )
    plt.text(20,20,str(best_epoch))
    plt.legend()
    plt.savefig(picture_File+'//'+'result.png')

def evaluate(model,testloader):
    model.eval()
    correct = 0
    for image,idx in testloader:
        image,idx = image.to(device),idx.to(device)
        with torch.no_grad():
            out = model(image)
            index = (torch.argmax(out)).item()
        if index is idx.item():
            correct+=1
    return correct/test_n_samples
train()

