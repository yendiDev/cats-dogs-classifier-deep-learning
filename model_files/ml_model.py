import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# load model 
model = keras.models.load_model('model_files/cat_dog_model.h5')
#model = keras.models.load_model('cat_dog_model.h5')

def predict_image(image_array):
    img_array = parse_image(image_array)
    prediction = model.predict([img_array])
    return prediction

def show_message(message):
    return message

# custom function for preprocessing image
def parse_image(array):
    image_array = np.expand_dims(array, axis=0) 
    return image_array

    
# # load image
# image = cv2.imread('dog_image.jpg', 1)
# print('image shape is: ', image)
# try:
#     img_resize = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
#     image_array = np.expand_dims(img_resize, axis=0) 
#     pred = predict_image(image)
#     print('The prediction made is: ', pred)

# except Exception as e:
#     print("The caught error is: ",str(e))

# #image_reshaped = cv2.resize(image, (300, 300), cv2.INTER_AREA)
# #preds = predict_image(image_reshaped)
# print('Final image size is: ', img_resize.shape)
