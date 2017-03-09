
from keras import backend as K
import image_loader as img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def show_image(path):
	image=mpimg.imread(path)
	imgplot = plt.imshow(image)
	plt.show()

height = 512
width = 512

base_image = K.variable(img.get_image('logan.jpg', height, width))
show_image('logan.jpg')


style_image1 = K.variable(img.get_image('styles/block.jpg', height, width))
show_image('styles/block.jpg')

style_image2 = K.variable(img.get_image('styles/ironman.jpg', height, width))
show_image('styles/ironman.jpg')

