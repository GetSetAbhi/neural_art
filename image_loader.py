import numpy as np
from PIL import Image


def get_image(path, height, width):
	pic = Image.open(path)
	pic = pic.resize((height, width))
	pic_array = np.asarray(pic, dtype='float32')
	pic_array = np.expand_dims(pic_array, axis=0)
	pic_array[:, :, :, 0] -= 103.939
	pic_array[:, :, :, 1] -= 116.779
	pic_array[:, :, :, 2] -= 123.68
	pic_array = pic_array[:, :, :, ::-1]
	return pic_array

def array_to_image(x1, height, width):
	x1 = x1.reshape((height, width, 3))
	# Convert back from BGR to RGB to display the image
	x1 = x1[:, :, ::-1]
	x1[:, :, 0] += 103.939
	x1[:, :, 1] += 116.779
	x1[:, :, 2] += 123.68
	x1 = np.clip(x1, 0, 255).astype('uint8')
	img_final = Image.fromarray(x1)
	return img_final