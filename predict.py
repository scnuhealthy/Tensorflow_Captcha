# Author: Kemo Ho
# This file to load some samples and predict

import tensorflow as tf
from load_data import *
import captcha_params_and_cfg
import captcha_cnn_model
import numpy as np

FLAGS = None

MAX_CAPTCHA = captcha_params_and_cfg.get_captcha_size()
CHAR_SET_LEN = captcha_params_and_cfg.get_char_set_len()

IMG_ROWS, IMG_COLS = captcha_params_and_cfg.get_height(), captcha_params_and_cfg.get_width()

x = tf.placeholder(tf.float32, [None,IMG_ROWS,IMG_COLS,1])
y_ = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
predict, keep_prob = captcha_cnn_model.deepnn(x)

predict = tf.reshape(predict,[-1,MAX_CAPTCHA,CHAR_SET_LEN])

X_predict,Y_predict = load_data_predict(tol_num = 100)
X_predict = X_predict.reshape(X_predict.shape[0],IMG_ROWS,IMG_COLS,1)
X_predict /=255
Y_predict = Y_predict.reshape(Y_predict.shape[0],MAX_CAPTCHA,CHAR_SET_LEN)

saver = tf.train.Saver()
with tf.Session() as sess:
	
	# load the trained model
	try:
		model_file=tf.train.latest_checkpoint(captcha_params_and_cfg.model_path)
		saver.restore(sess, model_file)
		print('loading model from %s' % model_file)
	except:
		print('No trained model!')
		exit()
	# predict and compare the results with the true label
	for i in range(X_predict.shape[0]):
		predict_result = sess.run(predict, feed_dict={x: [X_predict[i]], keep_prob: 1})
		predict_result = predict_result[0]
		print("Predict:",np.argmax(predict_result,1),"True label:",np.argmax(Y_predict[i],1))
