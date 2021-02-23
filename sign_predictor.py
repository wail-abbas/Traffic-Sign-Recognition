import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import expand_dims, nn


df_signnames = pd.read_csv('data/signnames.csv')


def predict_sign(sign_path, model):
    signnames = df_signnames['SignName']
    img = load_img(sign_path, target_size=(30, 30))
    # Image to Array
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0) # Create a batch
    # Predict the imag
    pred = model.predict(img_array)
    score = nn.softmax(pred[0])
    prediction  = "It's Probably the '{}' sign!" .format(signnames[np.argmax(score)])
    return prediction