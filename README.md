# Facial_Recognition
This is the repository for the machine learning final project of facial recognition.

#### Image_Processing

- First, you are supposed to create a document called <b>"photo"</b>, Then you need to divide each set of pictures of each person like following the format of

  <center><b>./photo/name/Your picture</b></center>

- Then you can create a directory for saving the processing photos called<b> "train"</b>

- After that, you can start to do the image processing by running the code

  <center><b>python image_processing.py</b></center>

#### Train

- First, you are supposed to download the pretrained data of Vgg model and you can search it from Google. Also, As you will download facenet_pytorch, you can get the pretrained data easily and place it into the directory of <b>"checkpoint"</b> or you can choose to not to use the pretrained model.

- Also, you are supposed to create two directories. one is for saving the model,<b>model</b>,the other is for saving the results for tuning the hyper parameter,<b>pictures</b>  

  <center><b>python face_recognition_my.py</b></center>

#### Predicted

- There are three different interfaces for you to do the prediction and one is real-time and others are not real-time.
- In the predicted.py, You will see video_demo1 and video_demo2, these are not real-time but it has higher accuracy. And video_demo 1 only can detect one person while video_demo_2 can detect all the person in your camera.



#### Hope you enjoy the facial recognition and it is all implemented by myself. If you have any problem, feel free to leave me a comment and I will make a reply as soon as possible.

