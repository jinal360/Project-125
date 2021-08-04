import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())

classes = ['A','B','C','D','E','F','G','H','I','I','J','K','L','M','N','O','P','Q',
'R','S','T','U','V','W','X','Y','Z']

nclasses = len(classes)

X_train,X_test,y_train,y_test = train_test_split(X,y , random_states = 9, train_size = 7500, test_size = 2500)

X_train_scale = X_train/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled,y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.conver('L')
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel,0,255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized/max_pixel)
    test_sample = np.array(image_bw_resized_inverted_scaled).redhape(1,660)
    test_prd = clf.perdict(test_sample)

    return test_prd[0]
