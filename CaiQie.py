import glob
import os
import random

import cv2

def caiqie1(img1,img2,label,l,train_path,num):
    x=1
    while (x <=num):
        i = random.randint(0, label.shape[0] -l-1)
        j = random.randint(0, label.shape[1] - l-1)
        im1 = img1[i:(i + l), j:(j + l)]
        im2 = img2[i:(i + l), j:(j + l)]
        label1 = label[i:(i + l), j:(j + l)]

            # cv2.imwrite("./data/shuguang/train/image1/"+str(i*l)+"_"+str(j*l)+".bmp",im3)
        cv2.imwrite(train_path + "/image1/" + str(i) + "_" + str(j) + ".jpg", im1)
        cv2.imwrite(train_path + "/image2/" + str(i) + "_" + str(j) + ".jpg", im2)

        cv2.imwrite(train_path + "/label/" + str(i) + "_" + str(j) + ".jpg", label1)
        x = x + 1

path="Italy"
# path=""
img1 = cv2.imread("./data/"+path+"/pred/image1/pre.tif")
img2 = cv2.imread("./data/"+path+"/pred/image2/post.tif")
label = cv2.imread("./data/"+path+"/pred/label/gt.bmp")
label=cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
h=img1.shape[0]
w=img1.shape[1]
l=16
num=100
h=int(h/l)
w=int(w/l)
train_path="./data/"+path+"/train"
caiqie1(img1,img2,label,l,train_path,num)
print("okok")

