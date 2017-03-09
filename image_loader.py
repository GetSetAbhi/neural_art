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