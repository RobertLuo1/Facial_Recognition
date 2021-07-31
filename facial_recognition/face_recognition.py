from facenet_pytorch import MTCNN,extract_face
import os
import torch
from torch import nn,optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset,Dataset
from PIL import Image,ImageFont,ImageDraw
import cv2
import face_model as fm
import numpy as np
import matplotlib.pyplot as plt
import datetime
model_File = r"./model.pth"
feature_File = r"./feature.pth"
test_File = r"./test"
mtcnn0 = MTCNN(image_size=240,min_face_size=40,keep_all=False)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True

dataset = datasets.ImageFolder('photo') # photos folder path
test_n_samples = len(dataset.samples)/6
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}# accessing names of peoples from folder names
names = list(idx_to_class.values())
classes = len(idx_to_class.keys())
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

def make_test_folder(testset):
    size = os.path.getsize(test_File)
    if size ==0:
        for i in range(int(test_n_samples)):
            img = testset[i][0]
            label = names[testset[i][1]]
            outfile = test_File+'//'+label+'.jpg'
            img.save(outfile)
    else:
        pass

datasets = train_val_dataset(dataset)
make_test_folder(datasets['test'])##为了之后的测试


def collate_fn(x):
    return x[0]
trainloader_recognition = DataLoader(datasets['train'],collate_fn=collate_fn,shuffle=True,batch_size=4)
trainloader_verification = DataLoader(datasets['train'],collate_fn=collate_fn)
testloader = DataLoader(datasets['test'],collate_fn=collate_fn,shuffle=True)
resnet = fm.InceptionResnetV1(classify=True,num_classes=len(idx_to_class.keys()),device=device,pre_trained=True).eval()


def train(model,trainloader):
    epochs = 30
    optim_para =[]
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.TripletMarginLoss()
    for para in model.parameters():
        para.requires_grad = False
    # for para in model.last_linear.parameters():
    #     para.requires_grad = True
    #     optim_para.append(para)
    # for para in model.last_bn.parameters():
    #     para.requires_grad = True
    #     optim_para.append(para)
    for para in model.logits.parameters():
        para.requires_grad = True
        optim_para.append(para)
    optimizer = optim.Adam(optim_para, lr=0.001)
    train_losses = []
    accuracy = []
    for e in range(epochs):
        running_loss = 0.0
        for image,idx in trainloader:
            face, prob = mtcnn0(image, return_prob=True)
            if face is not None and prob > 0.92:
                out = model(face.unsqueeze(0).to(device))
                loss = criterion(out, torch.tensor([idx]).to(device))
                loss.requires_grad_(True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))
        accuracy.append(test(model,testloader))
    plt.title("Train_loss and accuracy")
    plt.plot(train_losses, label='Training loss', )
    plt.plot(accuracy,label="accuracy",)
    plt.legend()
    plt.show()

def test(model,testloader):
    correct = 0
    for image,idx in testloader:
        face,prob = mtcnn0(image,return_prob=True)
        if face is not None and prob>0.92:
            out = model(face.unsqueeze(0).to(device))
            index = (torch.argmax(out)).item()
            if index is idx:
                correct+=1
    return correct/test_n_samples


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def predicted(predicted_model):
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("fail to grab frame, try again")
            break
        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
            for i, prob in enumerate(prob_list):
                if prob > 0.92:
                    out = predicted_model(img_cropped_list[i].unsqueeze(0).to(device))
                    index = torch.argmax(out,dim=1)
                    name = idx_to_class[index.item()]
                    box = boxes[i]
                # original_frame = frame.copy()  # storing copy of frame before drawing on it
                #     frame = cv2.putText(frame, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                #                         1, (0, 255, 0), 1, cv2.LINE_AA)
                    frame = cv2ImgAddText(frame,name,box[0],box[1],textColor=(0,255,0),textSize=20)
                    frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

                cv2.imshow("IMG", frame)

                k = cv2.waitKey(1)
                if k % 256 == 27:  # ESC
                    print('Esc pressed, closing...')
                    cam.release()
                    cv2.destroyAllWindows()
    cam.release()
    cv2.destroyAllWindows()

Resnet_recognition = fm.InceptionResnetV1(classify=True,num_classes=classes,device=device,pre_trained=False)
Resnet_recognition.load_state_dict(torch.load(model_File,map_location='cpu'))
Resnet_recognition.eval()#加载模型

def facial_recognition(test_path):
    image = Image.open(test_path)
    face,prob = mtcnn0(image,return_prob=True)
    if face is not None and prob>0.90:
        out = Resnet_recognition(face.unsqueeze(0).to(device))
        index = torch.argmax(out,dim=1)
        name = names[index.item()]
        return name+'.jpg'
    else:
        return names[np.random.randint(0,classes-1)]+'.jpg'


def facial_recognition_2(test_path):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(240),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    image = Image.open(test_path)
    face = trans(image)
    out = Resnet_recognition(face.unsqueeze(0).to(device))
    index = torch.argmax(out,dim=1)
    name = names[index.item()]
    return name+'.jpg'


Resnet_verification = fm.InceptionResnetV1(classify=False,device=device,pre_trained=True)
Resnet_verification.eval()
embedding_list = []
name_list =[]
def feature_extraction(name_list,embedding_list):
    for image,idx in trainloader_verification:
        face,prob = mtcnn0(image,return_prob=True)
        if face is not None and prob>0.92:
            emb = Resnet_verification(face.unsqueeze(0).to(device))
            embedding_list.append(emb.detach().numpy())
            name_list.append(idx)
    new_name_list = np.array(name_list)
    new_embedding_list = np.array(embedding_list)
    return new_embedding_list,new_name_list

# embedding_list,name_list = feature_extraction(name_list,embedding_list)
# data = [embedding_list, name_list]
# torch.save(data, feature_File) # saving data.pt file
# load_data = torch.load(feature_File)
# embedding_list = load_data[0]
# name_list = load_data[1]


def facial_verification_2(id,test_path):
    image = Image.open(test_path)
    face,prob  = mtcnn0(image,return_prob=True)
    if face is not None and prob>0.90:
        emb = Resnet_verification(face.unsqueeze(0).to(device))
        features = embedding_list[np.argwhere(name_list==id)]
        distance = np.zeros(features.shape[0])
        for idx, feature in enumerate(features):
            distance[idx] = torch.dist(emb,torch.from_numpy(feature)).item()
        if np.sum(distance) < (0.9*features.shape[0]):
            return "True"
        else:
            return "False"
    else:
        return 'False'

def facial_verification(id,test_path):
    image = Image.open(test_path)
    face, prob = mtcnn0(image, return_prob=True)
    if face is not None and prob > 0.90:
        out = Resnet_recognition(face.unsqueeze(0).to(device))
        index = torch.argmax(out, dim=1)
        if id==index.item():
            return 'True'
        else:
            return 'False'
    else:
        return 'False'

def video_demo_2():
    #0是代表摄像头编号，只有一个的话默认为0
    capture=cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()
    index=1
    while(True):
        ref,frame=capture.read()
        cv2.imshow("The webcam",frame)
        c= cv2.waitKey(30) & 0xff #等待30ms显示图像，若过程中按“Esc”退出
        if c==27:
            capture.release()
            break
        if c == ord('s'): #按下s键，进入下面的保存图片操作
            img = Image.fromarray(frame)
            img_cropped_list, prob_list = mtcnn(img, return_prob=True)
            if img_cropped_list is not None:
                boxes, _ = mtcnn.detect(frame)
                for i in range(boxes.shape[0]):
                    face = extract_face(frame,boxes[i]).numpy().transpose(1,2,0)
                    cv2.imwrite("./class_data_2/test_images/"+ "test_"+ str(index+i) + ".jpg", face)
                    print("-------------------------")
                    result_name=facial_recognition_2("./class_data_2/test_images/"+ "test_"+ str(index+i) + ".jpg")
                    print(result_name)
                index += boxes.shape[0]

        elif c == ord('q'):     #按下q键，程序退出
            break
def video_demo_1():
    #0是代表摄像头编号，只有一个的话默认为0
    capture=cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    index=1
    while(True):
        ref,frame=capture.read()
        cv2.imshow("The webcam",frame)
        c= cv2.waitKey(30) & 0xff #等待30ms显示图像，若过程中按“Esc”退出
        if c==27:
            capture.release()
            break
        if c == ord('s'):   #按下s键，进入下面的保存图片操作
            cv2.imwrite("./class_data/test_images/"+ "test_"+ str(index) + ".jpg", frame)
            print("-------------------------")
            result_name=facial_recognition("./class_data/test_images/"+ "test_"+ str(index) + ".jpg")
            print(result_name)
            index += 1

        elif c == ord('q'):     #按下q键，程序退出
            break

# video_demo_1()

# video_demo_2()

# starttime = datetime.datetime.now()
# right = 0
# wrong = 0
#
# for item in os.listdir("test"):
#     test_path = "test/" + item
#     if facial_recognition(test_path) == item:
#         right += 1
#     else:
#         wrong += 1
#
# accuracy = right / (right+wrong)
# endtime = datetime.datetime.now()
#
# print("人脸识别的考察结果：")
# print("人脸识别的准确率是:", accuracy)
# print("整个人脸识别的运行时间是：", (endtime-starttime).seconds, "s")


# tp = 0
# tn = 0
# fp = 0
# fn = 0
#
# for name in names:
#     test_path = "test/" + name + ".jpg"
#     for id in range(len(names)):
#         result = facial_verification(id, test_path)
#         if name == names[id] and result == "True":
#             tp +=1
#         elif name == names[id] and result == "False":
#             fn += 1
#         elif name != names[id] and result == "False":
#             tn += 1
#         else:
#             fp += 1
#
# print("人脸认证的考察结果:")
# print("精度:", tp/(tp+fp))
# print("回归率:", tp/(tp+fn))
# print("特异性:", tn/(tn+fp))
# print("F1值:", 2*tp/(2*tp+fp+fn))





#直接点debug就可以
# train(resnet,trainloader_recognition)
# print(test(resnet,testloader))
# torch.save(resnet.state_dict(),model_File)
# Resnet = fm.InceptionResnetV1(classify=True,num_classes=classes,device=device,pre_trained=False)
# Resnet.load_state_dict(torch.load(model_File,map_location='cpu'))
# Resnet.eval()
# predicted(Resnet_recognition)
