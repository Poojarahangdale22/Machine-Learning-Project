#!/usr/bin/env python
# coding: utf-8

# # Fashion Product Classifier
# 
# 
# 

# ![Image Alt Text]("C:\Fashion-models-character-on-transparent-background-PNG.png")

# In[1]:


from IPython.display import Image

# Specify the file path or URL of the image
image_path = "C:\Fashion-models-character-on-transparent-background-PNG.png"

# Display the image
Image(filename=image_path)


# ## Downloading the Data

# In[ ]:


get_ipython().system('pip install opendatasets')


# In[ ]:


import opendatasets as od


# In[ ]:


od.download("https://www.kaggle.com/paramaggarwal/fashion-product-images-small")


# In[ ]:


cd fashion-product-images-small/


# In[ ]:





# ## Image name with Extension

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("styles.csv",error_bad_lines=False)


# In[ ]:


df.head(10)


# In[ ]:


df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)


# In[ ]:


df.head(10)


# In[ ]:


df = df.sample(frac=1)


# In[ ]:


df.head(10)


# In[ ]:


df = df.sample(frac=1).reset_index(drop=True)


# In[ ]:


df.head(10)


# ## Preparing the ImageDataGenerator for training the model

# In[ ]:


from keras_preprocessing.image import ImageDataGenerator
batch_size = 256

image_generator = ImageDataGenerator(
    validation_split=0.2
)

training_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory="images",
    x_col="image",
    y_col="masterCategory",
    target_size=(60,80),
    batch_size=batch_size,
    subset="training"
)

validation_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory="images",
    x_col="image",
    y_col="masterCategory",
    target_size=(60,80),
    batch_size=batch_size,
    subset="validation"
)
classes = len(training_generator.class_indices)


# In[ ]:


classes


# ## Initializing the Model using Neural Network

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[ ]:


#Initializing our model
classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape = (60,80,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Conv2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Flatten())

classifier.add(Dense(units=32,activation = 'relu'))
classifier.add(Dense(units=64,activation = 'relu'))
classifier.add(Dense(units=128,activation = 'relu'))
classifier.add(Dense(units=256,activation = 'relu'))
classifier.add(Dense(units=512,activation = 'relu'))

classifier.add(Dense(units=7,activation = 'softmax'))

classifier.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])


# ## Fitting Data in our Neural Network

# In[ ]:


from math import ceil

classifier.fit_generator(
    generator=training_generator,
    steps_per_epoch=ceil(0.8 * (df.shape[0] / batch_size)),

    validation_data=validation_generator,
    validation_steps=ceil(0.2 * (df.shape[0] / batch_size)),

    epochs=5,
    verbose=1
)

loss, acc = classifier.evaluate_generator(validation_generator, steps=ceil(0.2 * (df.size / batch_size)))
print("\n%s: %.2f%%" % (classifier.metrics_names[1], acc * 100))


# ## Saving the Model for further use

# In[ ]:


classifier.save("model.h5")


# ## Testing on our data

# In[ ]:


import numpy as np
from keras.preprocessing import image

filename = "15970.jpg"

from keras.models import load_model
new_model = load_model('model.h5')
new_model.summary()

test_image = image.load_img("images/"+filename,target_size=(60,80))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = new_model.predict(test_image)
print(result)


# In[ ]:


val = np.argmax(result)
print(val)


# In[ ]:


my_dict = training_generator.class_indices
key_list = list(my_dict.keys()) 
val_list = list(my_dict.values()) 
print(key_list[val])


# ## Saving the Class IDs

# In[ ]:


import pickle
# save the model to disk
filename1 = 'key_list'
filename = 'val_list'
pickle.dump(key_list, open(filename1, 'wb'))
pickle.dump(val_list, open(filename, 'wb'))


# In[ ]:




