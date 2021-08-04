'''
函数说明: 
Author: hongqing
Date: 2021-08-03 14:50:45
LastEditTime: 2021-08-03 17:34:15
'''
import cv2
import mediapipe as mp
import os
import pickle
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pathfile="E://pose//"
savepath="E://pose//traindata//"
# For webcam input:
filenames=os.listdir(pathfile)
for files in filenames:
    if(files=='traindata'):
        continue
    jpgfile = pathfile+files
    jpglist= os.listdir(jpgfile)
    res = []
    for file in jpglist:
            if file.split('.')[1]=="jpg":
                img=cv2.imread(jpgfile+r'\\'+file)
                image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    res.append(results.multi_hand_landmarks)
                    # path_dir =  savepath+files
                    # folder = os.path.exists(path_dir)
                    # if not folder:
                    #     os.makedirs(path_dir) 
                    
                    # path_data = savepath+files+r'//'+file.split('.')[0]
                    path_data = savepath+files
                    with open(path_data, 'wb') as p:
                        pickle.dump(res,p)