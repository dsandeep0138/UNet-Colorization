from keras.models import Model
from keras.layers import Input, Reshape, Dense, Concatenate, Cropping2D, Activation, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.layers.core import Lambda
import keras.backend as K

class UNetClassifier(object):

	def __init__(self,
		     num_layers,
		     num_filters,
		     num_classes):
		self.num_layers = num_layers
		self.num_filters = num_filters
		self.num_classes = num_classes

	def build_model(self):

		# Input to the model
		inputs = Input(shape=(256, 256, 1), name = 'image_input')

		# U-Net Architecture:
		# Step 1: Input -> 3x3 conv with RELU (BN?) and filters f -> 3x3 conv with RELU (BN?)
		# Step 2: Max pooling with pool size (2,2)
		# Step 3: Repeat Step 1 and 2 with filters 2f until num_layers * f
		# Step 4: Now upsample and at each step filters divide by 2
		# Step 5: Output -> 3x3 conv with RELU -> 3x3 conv with relu
		# Step 6: Add conv with filters 2 and relu activation at the end
		# Step 7: Add all the skip connections

		output = inputs
		conv_layers = []

		# Downsampling layers
		for i in range(self.num_layers):
			conv1 = Conv2D(self.num_filters,
				       kernel_size=(3, 3),
				       strides=(1, 1),
				       padding='same',
				       activation='relu')(output)
			
			conv2 = Conv2D(self.num_filters,
				       kernel_size=(3, 3),
				       strides=(1, 1),
				       padding='same',
				       activation='relu')(conv1)
		
			conv2 = BatchNormalization()(conv2)
			conv_layers.append(conv2)	
	
			# Default value of strides is pool_size
			# This would halve the input.
			output = MaxPooling2D(pool_size=(2, 2))(conv2)

			self.num_filters *= 2

		# Bottleneck block
		bottleneck = output

		bnconv1 = Conv2D(self.num_filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(bottleneck)

		bnconv2 = Conv2D(self.num_filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(bnconv1)
		output = BatchNormalization()(bnconv2)

		# Upsampling layers
		for i in range(self.num_layers):
			self.num_filters = self.num_filters // 2

			up = UpSampling2D(size=(2, 2))(output)

			merge = Concatenate()([up, conv_layers.pop()])
		
			deconv1 = Conv2D(self.num_filters,
					 kernel_size=(3, 3),
					 strides=(1, 1),
					 padding='same',
					 activation='relu')(merge)

			deconv2 = Conv2D(self.num_filters,
					 kernel_size=(3, 3),
					 strides=(1, 1),
					 padding='same',
		         		 activation='relu')(deconv1)

			output = BatchNormalization()(deconv2)
	
		# Final softmax layer
		output = Conv2D(self.num_classes,
				kernel_size=(1, 1),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(output)

		output = Activation('softmax')(output)

		model = Model(input=inputs, output=output)

		model.summary()
		
		return model	
