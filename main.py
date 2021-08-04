'''
函数说明: 
Author: hongqing
Date: 2021-08-04 14:03:40
LastEditTime: 2021-08-04 14:44:29
'''
import logging
import cv2
import math
import time
import mediapipe as mp
import torch
from mymodel import Net
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
net = Net(4)
net.load_state_dict(torch.load('lastweight.ckpt'))
net.eval()
mycalcount = 60
count=0
# def load():
#     _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
#     pretrained_dict = torch.load(yolov4conv137weight)

#     model_dict = _model.state_dict()
#     # 1. filter out unnecessary keys
#     pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict)
#     _model.load_state_dict(model_dict)

def Normalize_landmarks(image, hand_landmarks):
  new_landmarks = []
  for i in range(0,len(hand_landmarks.landmark)):
    float_x = hand_landmarks.landmark[i].x
    float_y = hand_landmarks.landmark[i].y
    width = image.shape[1]
    height = image.shape[0]
    pt = mp_drawing._normalized_to_pixel_coordinates(float_x,float_y,width,height)
    new_landmarks.append(pt)
  return new_landmarks
  
hands = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=1)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("camera frame is empty!")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            normalized_landmarks = Normalize_landmarks(image, hand_landmarks)
            count+=1
            data = []
            try:
                if(count%mycalcount==0):
                    for index,i in enumerate(hand_landmarks.landmark):
                        if(index == 0):
                            firstx = i.x
                            firsty = i.y
                            continue
                        data.append((i.x - firstx)/(i.y-firsty))
                    data = torch.Tensor(data)
                    out = net(data)
                    num = torch.max(out, 0)[1].item()
                    status='{}'.format(num+1)
                cv2.putText(image,status,(40,410),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
            except:
                pass
    cv2.imshow('result', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break