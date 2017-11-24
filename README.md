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
Acquire massive capthca by hunman is unrealistic. Fortunately, we can easily generate the captcha by the python package **captcha**.See the [get_train_set.py](https://github.com/scnuhealthy/cnn_keras_captcha/blob/master/code/get_train_set.py) in details.

# Build the cnn model
I build the network with [keras](https://github.com/fchollet/keras), using the theano backend. I think use keras raher than theano to code is much more convenient.     
The network includes three convolution layers and two full-connected layers. Owing to CNN needs a large amount of samples to train, with the limitation of time and resources, I only use digital number as the char set of the captcha. So output of the model has 4*10 numbers, while is will have 4*62 numbers if the char set includes ppercase and lowercase letters.
```python
# 3 conv layer
model.add(Conv2D(nb_filters1, (kernel_size[0], kernel_size[1]), padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(nb_filters2, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(nb_filters3, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# Fully connected layer
model.add(Flatten())
model.add(Dense(1024*MAX_CAPTCHA))
model.add(Dense(512*MAX_CAPTCHA))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(MAX_CAPTCHA*CHAR_SET_LEN))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
```

# The training result    
I use 18000 training samples and 6000 test samples. After 64 epochs of training, the model obtains the correct rate of 97.91% on the test data.There is part of the result following:    

![test_result](https://github.com/scnuhealthy/cnn_keras_captcha/blob/master/picture/test_result.jpg)  
![4063](https://github.com/scnuhealthy/cnn_keras_captcha/blob/master/picture/9_4063.jpg)    
![7229](https://github.com/scnuhealthy/cnn_keras_captcha/blob/master/picture/15_7229.jpg)   
We can see the program successfully recognize "4063" , but fail "7229". The chat '7' and '2' are similar, maybe we need more data to train.

# Try yourself
## Environment
My Environment is Mac with Anaconda. Anaconda should install the package tensorflow,keras and capthca.
## Run my program
- get_captcha.py: Generate the sample captchas
- captcha_train.py: Train the model
- eval_test.py: Solve an image 
```
	python eval_test.py data/50_7136.png
```
It works pretty well in order to solve captcha created from other libraries such phpcaptcha or using letters and more than 4 characters :)