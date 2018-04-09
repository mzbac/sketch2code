import pickle
import cv2
import numpy as np

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def word_for_id(integer):
    word_index = load_obj('word2index')
    for word, index in word_index.items():
        if index == integer:
            return word
    return None

def id_for_word(word):
    word_index = load_obj('word2index')
    return word_index[word]

def resize_img(png_file_path):
        img_rgb = cv2.imread(png_file_path)
        img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
        img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
        resized = cv2.resize(img_stacked, (224,224), interpolation=cv2.INTER_AREA)
        bg_img = 255 * np.ones(shape=(224,224,3))
        bg_img[0:224, 0:224,:] = resized
        bg_img /= 255
        bg_img = np.rollaxis(bg_img, 2, 0)  
        return bg_img


