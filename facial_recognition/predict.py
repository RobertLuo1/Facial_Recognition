from facenet_pytorch import MTCNN,extract_face
import os
import torch
from torchvision import transforms
from PIL import Image,ImageFont,ImageDraw
import cv2
import face_model as fm
import numpy as np

model_File = r"./model"
mtcnn0 = MTCNN(image_size=240,min_face_size=40,keep_all=False)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = torch.load(model_File+"//"+"name.pth")
classes = len(names)
Resnet_recognition = fm.InceptionResnetV1(classify=True,num_classes=classes,device=device,pre_trained=False)
Resnet_recognition.load_state_dict(torch.load(model_File+'//resnet.pth',map_location='cpu'))
Resnet_recognition.eval()#加载模型

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
                    name = names[index.item()]
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
