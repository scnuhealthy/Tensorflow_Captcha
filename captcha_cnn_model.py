# Author: Kemo Ho
# THis file is to build the CNN network

import captcha_params_and_cfg
import tensorflow as tf

MAX_CAPTCHA = captcha_params_and_cfg.get_captcha_size()
CHAR_SET_LEN = captcha_params_and_cfg.get_char_set_len()

IMG_ROWS, IMG_COLS = captcha_params_and_cfg.get_height(), captcha_params_and_cfg.get_width()


nb_filters = (32,64)                   # number of filters
D_out = (512,MAX_CAPTCHA*CHAR_SET_LEN) # output_dim_fully_connected_layers
pool_size = (2, 2)                     # pool size
kernel_size = (3, 3)                   # kernel size
nb_pools = 2

FLAGS = None

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, IMG_ROWS*IMG_COLS)

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, MAX_CAPTCHA*CHAR_SET_LEN), with values
    equal to the logits of classifying the digit into one of MAX_CAPTCHA*CHAR_SET_LEN classes 
	(if CHAR_SET is digital number and the number of elements in a captcha is one, the classes will be 
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, assuming that images have been 
  # transformed into grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, IMG_ROWS, IMG_COLS, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([kernel_size[0], kernel_size[1], 1, nb_filters[0]])
    b_conv1 = bias_variable([nb_filters[0]])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, nb_filters[0], nb_filters[1]])
    b_conv2 = bias_variable([nb_filters[1]])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, maps this to 1024 features.
  with tf.name_scope('fc1'):
    rows = IMG_ROWS//(pool_size[0]**nb_pools)
    print(rows)
    cols = IMG_COLS//(pool_size[1]**nb_pools)
    W_fc1 = weight_variable([rows * cols * nb_filters[-1], D_out[0]])
    b_fc1 = bias_variable([D_out[0]])

    h_pool2_flat = tf.reshape(h_pool2, [-1, rows*cols*nb_filters[-1]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to MAX_CAPTCHA*CHAR_SET_LEN classes, one for each situation
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([D_out[0], D_out[1]])
    b_fc2 = bias_variable([D_out[1]])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, pool_size[0], pool_size[1], 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
