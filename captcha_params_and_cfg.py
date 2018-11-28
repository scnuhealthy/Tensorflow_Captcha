# Author: Kemo Ho
# This file is to define the paths, captcha parameters and other configuration

import os

# char set
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# the number of elements in a captcha
MAX_CAPTCHA = 4

# the size of the captcha image
WIDTH = 120
HEIGHT= 60


def get_char_set():
	return number+alphabet

def get_char_set_len():
	return len(get_char_set())

def get_captcha_size():
	return MAX_CAPTCHA

def get_y_len():
	return MAX_CAPTCHA*get_char_set_len()

def get_width():
    return WIDTH

def get_height():
    return HEIGHT

model_path = './model'
model_tag = 'captcha4.model'
save_model = os.path.join(model_path, model_tag) # the path to save the trained model
data_path = 'data3'                            # the data path 
