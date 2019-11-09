from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Cropping2D, Activation, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D

class UNetRegressor(object):

	def __init__(self, num_layers, num_filters):
		self.num_layers = num_layers
		self.num_filters = num_filters

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

		conv11 = Conv2D(64,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(inputs)

		conv12 = Conv2D(64,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(conv11)

		conv12_crop = conv12 #Cropping2D(cropping=((42, 42), (42, 42)))(conv12)
		
		# Default value of strides is pool_size and this would halve the input.
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)

		conv21 = Conv2D(128,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(pool1)

		conv22 = Conv2D(128,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(conv21)

		conv22_crop = conv22 #Cropping2D(cropping=((17, 17), (17, 17)))(conv22)
		
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)

		conv31 = Conv2D(256,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(pool2)

		conv32 = Conv2D(256,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(conv31)

		conv32_crop = conv32 #Cropping2D(cropping=((4, 5), (4, 5)))(conv32)

		pool3 = MaxPooling2D(pool_size=(2, 2))(conv32)

		# Bottleneck block
		bottleneck = pool3

		bnconv1 = Conv2D(512,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(bottleneck)

		bnconv2 = Conv2D(512,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(bnconv1)
	
		up1 = UpSampling2D(size=(2,2))(bnconv2)

		merge1 = Concatenate()([up1, conv32_crop])

		deconv31 = Conv2D(256,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(merge1)

		deconv32 = Conv2D(256,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(deconv31)
	
		up2 = UpSampling2D(size=(2,2))(deconv32)

		merge2 = Concatenate()([up2, conv22_crop])
		
		deconv21 = Conv2D(128,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(merge2)

		deconv22 = Conv2D(128,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(deconv21)

		up3 = UpSampling2D(size=(2,2))(deconv22)

		merge3 = Concatenate()([up3, conv12_crop])

		deconv11 = Conv2D(64,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation='relu')(merge3)

		deconv12 = Conv2D(64,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(deconv11)

		output = Conv2D(2,
				kernel_size=(1, 1),
				strides=(1, 1),
				padding='same',
		         	activation='relu')(deconv12)

		model = Model(input=inputs, output=output)

		model.summary()
		
		return model	
