# Tensorflow_Captcha
Recognize the captchas(numbers and alphbet) with tensorflow using CNN

# What is the capthca
Captcha is image  generated by chars randomly and proper voice to avoid the attack from the robot.   
![9_9530](https://cloud.githubusercontent.com/assets/5999851/21477290/41594870-cb7b-11e6-8d3e-4e0c24af37f1.jpg)    
The picture above is a common captcha,including four digital numbers.  
#How to recognize the captcha
There three main methods:
> * find the loopholes of the captcha
> * spilt the captcha into single char, and then recognize the chars
> * regard the captha as the whole, and recognzie it directly   

Tesseract OCR and OpenCV use the second method. But the captcha is more and more complex now. There is a common phenomenon that the chars in the captcha interlace with one another, so the second method gets a low accuracy. This program focuses on the third method. 

# Generate the captcha to train
Acquire massive capthca by hunman is unrealistic. Fortunately, we can easily generate the captcha by the python package **captcha**.See the [get_train_set.py](https://github.com/scnuhealthy/Tensorflow_Captcha/blob/master/get_train_set.py) in details.

# Build the cnn model with tensorflow
I have another [project](https://github.com/scnuhealthy/cnn_keras_captcha) to achieve this with keras. In this project, I use tensorflow, for its scalability and prospect. The network includes two convolution layers and two full-connected layers. The charset includes digital numbers and the alphabet(only lowercase letters).
```python
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

# Fully connected layer 1 
with tf.name_scope('fc1'):
rows = IMG_ROWS//(pool_size[0]**nb_pools)
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

# Map the feactures to the classes
with tf.name_scope('fc2'):
W_fc2 = weight_variable([D_out[0], D_out[1]])
b_fc2 = bias_variable([D_out[1]])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```
# File structure
- captcha_params_and_cfg.py: This file is to define the paths, captcha parameters and other configuration
- get_train_set.py: This file is to get the data set for training
- load_data.py: This file is to load the data from the data set
- predict.py: This file to load some samples and predict
- train.py: This file is to train the model with the networks builded in captcha_cnn_model.py


# The result    
I use 60000 training samples and 10000 test samples. At the 23 epoch of training, the trainging error is almost zero. After 50 epochs of training, the model obtains the correct rate of 96.95% on the test data.There is part of the result following,where labels 0-9 represent digital numbers and labels 10-35 represent the lowercase letters:   

![trainging_accuracy](https://github.com/scnuhealthy/Tensorflow_Captcha/blob/master/traiing_accuracy.png)  
![predict_result](https://github.com/scnuhealthy/Tensorflow_Captcha/blob/master/predict_result.png)

# Why I rewrite this project with tensorflow?
Keras is not flexible. We can define new loss function, activation function in Tensorflow. The one reason is to practice. And the other is the captcha regonize projects in github done with tensorflow are not clear enough. And I have the confident that my code is easy for beginners to understand.

# Try yourself
## Environment
My Environment is Windows with Anaconda. Anaconda should install the package tensorflow and capthca.

# Pretrained model Download
https://1drv.ms/f/s!ArJ7K9H1C8nbiA2N26KTqPxea7nU

## Run my program
- get_train_set.py: Generate the sample captchas
- train.py: Train the model
- predict.py: Solve an image 
If your computer has not enough memory, you can just let char set only including digital numbers and reduce the size of the training set.
It works pretty well in order to solve captcha created from other libraries such phpcaptcha or using letters and more than 4 characters :)
