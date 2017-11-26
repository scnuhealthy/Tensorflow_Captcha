from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import captcha_params_and_cfg

# height_p = 60
# width_p = 120

# generate  the captcha text randomly from the char lists above
def random_captcha_text(char_set=captcha_params_and_cfg.get_char_set(), captcha_size=captcha_params_and_cfg.get_captcha_size()):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text
 
# generate the captcha text and image and save the image 
def gen_captcha_text_and_image(i):
	image = ImageCaptcha(width=captcha_params_and_cfg.get_width(), height=captcha_params_and_cfg.get_height(), font_sizes=[50])
 
	captcha_text = random_captcha_text()
	captcha_text = ''.join(captcha_text)

	path = captcha_params_and_cfg.data_path+'/'
	if os.path.exists(path) == False: # if the folder is not existed, create it
		os.mkdir(path)
                
	captcha = image.generate(captcha_text)

	# naming rules: num(in order)+'_'+'captcha text'.include num is for avoiding the same name
	image.write(captcha_text, path+str(i)+'_'+captcha_text + '.png') 
 
	captcha_image = Image.open(captcha)
	captcha_image = np.array(captcha_image)
	return captcha_text, captcha_image
 
if __name__ == '__main__':

        
        for i in range(24000):     
                text, image = gen_captcha_text_and_image(i)

        # show the image
        '''
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
	plt.imshow(image)
 
	plt.show()
        '''
