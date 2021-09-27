from facenet_pytorch import MTCNN,extract_face
import os
import torch
from torchvision import datasets
from PIL import Image
import numpy as np

train_File = r"./train"
mtcnn0 = MTCNN(image_size=240,min_face_size=40,keep_all=False)
dataset = datasets.ImageFolder('photo') # photos folder path
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}# accessing names of peoples from folder names
names = list(idx_to_class.values())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(dataset):
    images = []
    before = 0
    for i in range(len(dataset.samples)):
        img = dataset[i][0]
        label = dataset[i][1]
        if before != label and len(images) != 0:
            os.mkdir(train_File+'//'+names[before])
            for j in range(len(images)):
                image = images[j]
                outfile = train_File + '//' + names[before] +'//'+str(j)+ '.jpg'
                image.save(outfile)
            images.clear()
        face,prob  =mtcnn0(img,return_prob=True)
        if face is not None and prob>0.92:
            boxes,_ = mtcnn0.detect(img)
            img_face = extract_face(img, boxes[0]).numpy().transpose(1,2,0)
            img_pil = Image.fromarray(np.uint8(img_face))
            images.append(img_pil)
            before = label
    print('Finish')

preprocess(dataset)