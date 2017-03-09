import numpy as np
from keras import backend as K
from PIL import Image
import time
from keras.models import Model
from keras.applications import vgg16
from scipy.optimize import fmin_l_bfgs_b
import image_loader as img


height = 512
width = 512

base_image = K.variable(img.get_image('logan.jpg', height, width))
#style_image1 = K.variable(img.get_image('styles/block.jpg', height, width))
style_image2 = K.variable(img.get_image('styles/ironman.jpg', height, width))
combination_image = K.placeholder((1, height, width, 3))

#combine the3 images into a single Keras tensor
'''input_tensor = K.concatenate([base_image,
	style_image1,
	style_image2, 
	combination_image], axis = 0)'''

input_tensor = K.concatenate([base_image,
	style_image2, 
	combination_image], axis = 0)

#build the VGG16 network with our 3 images as input

model = vgg16.VGG16(input_tensor = input_tensor,
	weights = 'imagenet',
	include_top = False)

print('model loaded')

layers = dict([(layer.name, layer.output) for layer in model.layers])

content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

loss = K.variable(0.)

def content_loss(content, combination):
    return K.sum(K.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
#combination_features = layer_features[3, :, :, :]
combination_features = layer_features[2, :, :, :]


loss += content_weight * content_loss(content_image_features,combination_features)

def gram_matrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram


def style_loss(style, combination):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = height * width
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# Removable
def style_loss_two(style1, style2, combination):
	S1 = gram_matrix(style1)
	S2 = gram_matrix(style2)
	C = gram_matrix(combination)
	channels = 3
	size = height * width
	return backend.sum(backend.square(S1 - C) + backend.square(S2 - C) ) / (4. * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv2', 'block2_conv2',
'block3_conv3', 'block4_conv3',
'block5_conv3']

# Apply Style image 1
for layer_name in feature_layers:
	layer_features = layers[layer_name]
	style_features = layer_features[1, :, :, :]

	'''#removable
				style_features2 = layer_features[2, :, :, :]
				combination_features = layer_features[3, :, :, :]
				sl = style_loss_two(style_features,style_features2, combination_features)'''

	combination_features = layer_features[2, :, :, :]
	sl = style_loss(style_features,combination_features)

	loss += (style_weight / len(feature_layers)) * sl


def total_variation_loss(x):
	a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
	b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)

outputs = [loss]
outputs += grads
f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
	x = x.reshape((1, height, width, 3))
	outs = f_outputs([x])
	loss_value = outs[0]
	grad_values = outs[1].flatten().astype('float64')
	return loss_value, grad_values

class Evaluator(object):

	def __init__(self):
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 10

for i in range(iterations):
	print('Start of iteration', i)
	start_time = time.time()
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
	                                 fprime=evaluator.grads, maxfun=20)
	print('Current loss value:', min_val)
	end_time = time.time()
	print('Iteration %d completed in %ds' % (i, end_time - start_time))
	x = x.reshape((height, width, 3))
	x = x[:, :, ::-1]
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	x = np.clip(x, 0, 255).astype('uint8')
	img_final = Image.fromarray(x)
	name = 'result' + str(i) + '.png'
	img_final.save('output/'+name)