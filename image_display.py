import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def show_image(path):
	image=mpimg.imread(path)
	imgplot = plt.imshow(image)
	plt.show()

show_image('logan.jpg')

show_image('styles/block.jpg')

show_image('styles/ironman.jpg')

