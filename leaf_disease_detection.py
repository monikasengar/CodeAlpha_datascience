#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[5]:


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = r"C:\Users\monik\Downloads\sample_leaf_disease1\New Plant Diseases Dataset(Augmented)"
width=256
height=256
depth=3


# In[6]:


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[7]:


image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# In[8]:


image_size = len(image_list)


# In[9]:


label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)


# In[10]:


print(label_binarizer.classes_)


# In[11]:


np_image_list = np.array(image_list, dtype=np.float16) / 225.0


# In[12]:


print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 


# In[13]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# In[14]:


model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))


# In[15]:


model.summary()


# In[17]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example data (assuming x_train and y_train are already defined)
x_train = np.random.rand(1000, 20)  # 1000 samples, 20 features
y_train = np.random.randint(0, 2, (1000, 1))  # Binary labels

# Initial learning rate and number of epochs
INIT_LR = 0.001
EPOCHS = 440

# Create a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# Initialize the optimizer with the learning rate schedule
opt = Adam(learning_rate=lr_schedule)

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))  # 20 input features
model.add(Dense(1, activation='sigmoid'))  # Final layer for binary classification

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] training network...")
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)


# In[18]:


# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

# Assuming x_test and y_test are defined
x_test = np.random.rand(200, 20)  # 200 samples, 20 features
y_test = np.random.randint(0, 2, (200, 1))  # Binary labels

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')


# In[19]:


import matplotlib.pyplot as plt

# Extract metrics from history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plot training and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

# Plot training and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[20]:


print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# In[21]:


from keras.models import load_model

# Save the model to disk
print("[INFO] Saving model...")
model.save('cnn_model.h5')

# If you want to load the model later, use:
# model = load_model('cnn_model.h5')


# In[22]:


import tensorflow as tf
y_pred = model.predict(x_test)


# In[24]:


print("Originally : ",label_binarizer.classes_[np.argmax(y_test[1])])
print("Predicted : ",label_binarizer.classes_[np.argmax(y_pred[1])])


# In[ ]:




