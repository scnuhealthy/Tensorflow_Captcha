import tensorflow as tf
from load_data import *
import captcha_params_and_cfg
import captcha_cnn_model

FLAGS = None

MAX_CAPTCHA = captcha_params_and_cfg.get_captcha_size()
CHAR_SET_LEN = captcha_params_and_cfg.get_char_set_len()

IMG_ROWS, IMG_COLS = captcha_params_and_cfg.get_height(), captcha_params_and_cfg.get_width()

def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys 

# Import data
(X_train, Y_train), (X_test, Y_test) = load_data(tol_num = 48000,train_num = 40000)
X_train = X_train.reshape(X_train.shape[0],IMG_ROWS,IMG_COLS,1)
X_test = X_test.reshape(X_test.shape[0],IMG_ROWS,IMG_COLS,1)[:100]
X_train /=255
X_test /=255
Y_test = Y_test[:100]

batch_size = 100

x = tf.placeholder(tf.float32, [None,IMG_ROWS,IMG_COLS,1])


y_ = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])

y_conv, keep_prob = captcha_cnn_model.deepnn(x)

label = tf.reshape(y_,[-1,MAX_CAPTCHA,CHAR_SET_LEN])
y_conv = tf.reshape(y_conv,[-1,MAX_CAPTCHA,CHAR_SET_LEN])

# Define loss and optimizer
with tf.name_scope('loss'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                          logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
  optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(label, 2))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)


saver = tf.train.Saver(max_to_keep = 1)
save_model = captcha_params_and_cfg.save_model
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	step = 0
	try:
		model_file=tf.train.latest_checkpoint(captcha_params_and_cfg.model_path)
		saver.restore(sess, model_file)
		print('loading model from %s' % model_file)
		step = int(model_file.split('-')[-1])+1
	except:
		pass
	j = 0
	for i in range(20):
		for batch_xs,batch_ys in generatebatch(X_train,Y_train,Y_train.shape[0],batch_size): 
			sess.run([optimizer,cross_entropy],feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.95})
			if j % 40 ==0:
				train_accuracy = accuracy.eval(feed_dict={
					x: batch_xs, y_:batch_ys, keep_prob: 1.0})
				print('step %d, training accuracy %g' % (step, train_accuracy))
			j +=1
		print('Saving model into %s-%d'% (save_model,step))
		saver.save(sess, save_model,global_step = step)
		step +=1
    

	print('test accuracy %g' % accuracy.eval(feed_dict={
		x: X_test, y_: Y_test, keep_prob: 1.0}))


