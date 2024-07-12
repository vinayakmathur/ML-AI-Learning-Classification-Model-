import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = '/content/PetImages/PetImages'
categories = ['Dog','Cat']
for category in categories:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap = "gray")
        plt.show()
        break
    break

print(img_array.shape)
Img_size = 70
new_array = cv2.resize(img_array,(Img_size,Img_size))
plt.imshow(new_array,cmap = "gray")
plt.show()

train_data = []
def create_train_data():
  DATADIR = '/content/PetImages/PetImages'
  categories = ['Dog','Cat']
  for category in categories:
      path = os.path.join(DATADIR,category)
      class_num = categories.index(category)
      for img in os.listdir(path):
            try:
              img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
              new_array = cv2.resize(img_array,(Img_size,Img_size))
              train_data.append([new_array,class_num])
            except Exception as e:
              pass
create_train_data()

x = []
y = []

for features,label in train_data:
   x.append(features)
   y.append(label)
x  = np.array(x).reshape(-1,Img_size,Img_size,1)
import pickle
pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()
pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
X = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
X = X/255.0
model = Sequential()
model.add( Conv2D(64,(3,3),input_shape = X.shape[1:])  )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add( Conv2D(64,(3,3)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss = "binary_crossentropy",optimizer = "adam",metrics = ["accuracy"])
import numpy as np
y = np.array(y)
model.fit(X,y,batch_size = 32, epochs = 5,validation_split  = 0.1)
model.save("CNN-Predict.Model")
