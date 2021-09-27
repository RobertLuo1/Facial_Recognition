import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,path):
        super(MyDataset,self).__init__()
        self.data = []
        self.labels = []
        for dirpaths, dirnames, filenames in os.walk(path):
            for i in range(len(dirnames)):
                for ds, ns, fs in os.walk(path + '//' + dirnames[i]):
                    for file in fs:
                        img = Image.open(path + '//' + dirnames[i] + "//" + file)
                        self.data.append(img)
                        self.labels.append(i)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(240),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = tf(self.data[idx])
        l = self.labels[idx]
        return image,l