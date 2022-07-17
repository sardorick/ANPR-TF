from fileinput import filename
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pytesseract as pt
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# load the trained model
model = tf.keras.models.load_model('./static/models/trained_object_detection.h5')


# detect the number plate coordinates
def object_detection(path, filename):
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(224, 224))
    image_arr = img_to_array(image1)/255.0 # convert into array and normalize the output
    # size of array
    h, w, d = image.shape
    test_arr = image_arr.reshape(1, 224, 224, 3)
    # predictions
    coords = model.predict(test_arr)
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounding rectangle
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    cv2.rectangle(image, pt1, pt2, (0, 255, 2), 3)
    # convert rgb into bgr
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./static/predict/{filename}', img_bgr)
    return coords

# read the image using pytesseract
def OCR(path, filename):
    img = np.array(load_img(path))
    cods = object_detection(path, filename)
    xmin, xmax, ymin, ymax = cods[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)
    text = pt.image_to_string(roi)
    print(text)
    return text