from sklearn.datasets import load_files  
import numpy as np
import pandas as pd
from keras.utils import np_utils
import cv2
import urllib
import pdf2image

import requests

import tensorflow as tf
import keras, os
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, MaxPool2D, AveragePooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.constraints import max_norm
from numpy import expand_dims
from io import BytesIO
from PIL import Image
from tabulate import tabulate

import os, io
from google.cloud import vision_v1
import urllib.request
from PIL import Image
from pdf2image import convert_from_path

import string
from datetime import datetime

import re
import os

import PyPDF2

import matplotlib.pyplot as plt

import glob
from matplotlib import pyplot as plt
import cv2
import os
from IPython import display
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

# !export GOOGLE_APPLICATION_CREDENTIALS = json_file

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_file

client = vision_v1.ImageAnnotatorClient()

import os
from pdf2image import convert_from_path

pdf_dir = pdf_dir
os.chdir(pdf_dir)
lista = []

for pdf_file in os.listdir(pdf_dir):

    if pdf_file.endswith(".pdf"):
        
        try:
            pages = convert_from_path(pdf_file, 500,poppler_path=poppler_path)
            pdf_file = pdf_file[:-4]

            for page in pages:

                page.save("splitted_images/%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")
        
        except:
            lista.append(pdf_file)


def iterate_data(start_inx, dataset_path, save_path):
    files = glob.glob(dataset_path+"/*")
    
    for inx, file in enumerate(files):
        try:    
            if inx<start_inx:
                continue
                
            print(inx)
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(7,7 ))
            plt.imshow(image)
            plt.axis('off')
            
            plt.show()
            chosen_class = input()

            save_path_class = os.path.join(save_path, chosen_class)
            if not os.path.isdir(save_path_class):
                os.makedirs(save_path_class)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            save_file_path = os.path.join(save_path_class, os.path.basename(file))
            cv2.imwrite(save_file_path, image)
            plt.clf()
            display.clear_output(wait=True)
        except Exception as ex:
            pass
        
dataset_path = dataset_path # directorium with images (unlabelled data)
save_path = save_path  # Folder for saving labels 
start_inx = 0 # the start index

iterate_data(start_inx, dataset_path, save_path)


### Building the model ###

def load_image(path):
  image = cv2.imread(path).astype('float32')
  image = preprocess_input(cv2.resize(image, dsize=(224,224)))
  return image

def image_process(files):
  vector = []
  for file in files:
    vector.append(load_image(file))
  return np.array(vector)

def load_dataset(path):
    data = load_files(path)
    color_files = np.array(data['filenames'])
    color_targets = np_utils.to_categorical(np.array(data['target']), 4)
    color_names = data['target_names']
    return color_files, color_targets, color_names

# load whole dataset
train_files, train_targets, train_names = load_dataset(train_dataset_path)
val_files, val_targets, val_names = load_dataset(test_dataset_path)

# print(tabulate([['Train', len(train_files), train_targets.shape, len(train_names)],
#                 ['Validation', len(val_files), val_targets.shape, len(val_names)], 
#                 ], headers=['Dataset', 'No. of files', 'Target shape', 'No. of classes']))

train_X = image_process(train_files)
# train_X.shape

val_X = image_process(val_files)
# val_X.shape

class_names_processed = list(map(lambda x : x, train_names))

# print(class_names_processed)


# Plotting the images with the labels
fig = plt.figure(figsize=(20,20))

rows = 5
columns = 4

for i in range(rows*columns):
    ax1 = fig.add_subplot(rows, columns,(i+1)) 
    # show the image
    image = cv2.imread(train_files[i])
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb, cmap='gray')

    title = "Class: "+class_names_processed[np.argmax(train_targets[i])]
    plt.title(title)

# Building the generator
def generator(X_samples, Y_samples, batch_size=50):
    """
    Lazy batch train/validation generator for memory efficiency
    """
    while True:
      for offset in range(0, len(X_samples), batch_size):
        batch_samplesX = X_samples[offset:offset+batch_size]
        batch_samplesY = Y_samples[offset:offset+batch_size]
        new_batch = []
        for image in batch_samplesX:
          image_resized = cv2.resize(image, dsize=(224,224))
          new_batch.append(image_resized)
        X_train = np.array(new_batch)
        y_train = batch_samplesY
        yield X_train, y_train
train_generator = generator(train_X, train_targets, batch_size=50)
test_generator =  generator(val_X, val_targets, batch_size=50)


# using VGG16 pretrained model
from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(include_top=False, input_shape = (224,224,3), weights='imagenet')

layer_1 = Dense(256, activation='relu')(model.output)
pool = GlobalAveragePooling2D()(layer_1)
layer_2 = Dense(128, activation='relu')(pool) # Added another layer with average
layer_3 = Dense(64, activation='relu')(layer_2)
drop_1 = Dropout(0.3)(layer_3)
layer_4 = Dense(16, activation='relu')(drop_1)
batch_norm = BatchNormalization()(layer_4)
drop_2 = Dropout(0.4)(batch_norm)
output = Dense(4, activation='softmax')(drop_2)

model = Model(inputs = model.inputs, outputs = output)

# model.summary()

# Using learning rate of 0.0001 with the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Generating more data for bigger accuracy
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen.fit(train_X)

# Training the model
anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-5)
checkpoint = ModelCheckpoint('best_model_4_classes_version_3.pt', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# model fit code
history = model.fit(datagen.flow(train_X, train_targets),
          validation_data = test_generator,
          steps_per_epoch = len(train_X)/50,
          validation_steps = len(val_X)/50,
          epochs=50, #added 5 epochs more
          callbacks=[anne,checkpoint],
          batch_size = 100)

# Prediction function
def get_prediction(image):
    image = np.expand_dims(image, axis=0)
    prediction = best_model.predict(image)
    predicted_class = np.argmax(prediction)
    return class_names_processed[predicted_class]

# get_prediction(load_image(some_image))

# Using the model for PDF files, this is a function that separates the pages in the PDF file and runs the model through each of them in order to give the prediction
# Also I am using Google Cloud OCR model through the API 
def sample_batch_annotate_files_url(file_path_url):
    import requests 
    """Perform batch file annotation."""
    rxcountpages = re.compile(r"/Type\s*/Page([^s]|$)", re.MULTILINE|re.DOTALL)
    client = vision_v1.ImageAnnotatorClient()

    response = requests.get(file_path_url)
    pdf_file = io.BytesIO(response.content) # response being a requests Response object
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    number = pdf_reader.numPages
    
    scrape = urllib.request.urlopen(file_path_url)  # for external files
    pil_images = pdf2image.convert_from_bytes(scrape.read(), dpi=200, 
                 output_folder=None, first_page=None, last_page=None,
                 thread_count=1, userpw=None,use_cropbox=False, strict=False,
                 poppler_path=poppler_path,)
    
    list_predicted_classes = []

    for page in pil_images:

        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        image_resized = preprocess_input(cv2.resize(image, dsize=(224,224)))
        
        predicted_class = get_prediction(image_resized)
        
        list_predicted_classes.append(predicted_class)
        
    print(list_predicted_classes)
    
    # Supported mime_type: application/pdf, image/tiff, image/gif
    mime_type = "application/pdf"
    with urllib.request.urlopen(file_path_url) as r:
        content = r.read()
        print ("Content: "+str(content[0]))

    input_config = {"mime_type": mime_type, "content": content}
    features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]

    while number > 0:
        i = 0
        batch = 5
        counter = 0
        page = 1
        j = 1
        lista = []
        for k in range(round(abs(number/5 + 0.5))):
            if number > 5:
                client = vision_v1.ImageAnnotatorClient()
                mime_type = "application/pdf"
#                 features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]
                pages=[]
                for page in range(j,batch+1):
                    pages.append(page)
                j+=5
                batch+=5
                requests = [{"input_config": input_config, "features": features, "pages": pages}]
                response = client.batch_annotate_files(requests=requests)
                for image_response in response.responses[0].responses:
                    counter += 1
                    print(f'Page: {counter}')
                    text_lista = image_response.full_text_annotation.text
                    lista.append(text_lista)
                    print(text_lista)
                number = number - 5
            else:
                client = vision_v1.ImageAnnotatorClient()
                mime_type = "application/pdf"
                features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]
                pages=[]
                for page in range(j,j+number):
                    pages.append(page)
                requests = [{"input_config": input_config, "features": features, "pages": pages}]
                response = client.batch_annotate_files(requests=requests)
                for image_response in response.responses[0].responses:
                    counter += 1
                    print(f'Page: {counter}')
                    text_lista = image_response.full_text_annotation.text
                    lista.append(text_lista)
                    print(text_lista)
                number = number - 5
                return list_predicted_classes, lista