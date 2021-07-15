import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, glob
from sklearn.model_selection import train_test_split

def resize_images(img_path):    # Resizing to 28x28
    
	images=glob.glob(img_path + "/*.jpg")  
    
	print(len(images), " images to be resized.")
    
	target_size=(28,28)
	for img in images:
		old_img=Image.open(img)
		new_img=old_img.resize(target_size,Image.ANTIALIAS)
		new_img.save(img, "JPEG")
    
	print(len(images), " images resized.")

def load_data(img_path, number_of_data=600):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    return imgs, labels, idx

# 각 폴더의 path
image_rock_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/rock"
image_paper_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/paper"
image_scissor_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/scissor"
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/"
image_rock_test_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/rock"
image_scissor_test_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/scissor"
image_paper_test_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/paper"
image_test_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test"

# 가위, 바위, 보 resizing
resize_images(image_rock_path)
resize_images(image_paper_path)
resize_images(image_scissor_path)
resize_images(image_rock_test_path)
resize_images(image_scissor_test_path)
resize_images(image_paper_test_path)

(x_train, y_train, idx)=load_data(image_dir_path)
print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
(x_test, y_test, idx)=load_data(image_test_path)
print("학습데이터(x_test)의 이미지 개수는", idx,"입니다.")

n_channel_1 = 16
n_channel_2 = 32
n_dense=128
n_train_epoch=10

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Normalization
x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

# Reshaping
x_train_reshaped=x_train_norm.reshape( -1, 28, 28, 3)  # 데이터갯수에 -1을 쓰면 reshape시 자동계산됩니다.
x_test_reshaped=x_test_norm.reshape( -1, 28, 28, 3)

model.fit(x_train_reshaped, y_train, epochs=n_train_epoch)

# 모델 시험
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
