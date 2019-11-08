from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, BatchNormalization,MaxPooling2D, UpSampling2D

class UNetRegressor(object):

	def __init__(self, num_layers, num_filters):
		self.num_layers = num_layers
		self.num_filters = num_filters

	def build_model(self):

		# Input to the model
		inputs = Input(shape=(256, 256, 3), name = 'image_input')

		# U-Net Architecture:
		# Step 1: Input -> 3x3 conv with RELU (BN?) and filters f -> 3x3 conv with RELU (BN?)
		# Step 2: Max pooling with pool size (2,2)
		# Step 3: Repeat Step 1 and 2 with filters 2f until num_layers * f
		# Step 4: Now upsample and at each step filters divide by 2
		# Step 5: Output -> 3x3 conv with RELU -> 3x3 conv with relu
		# Step 6: Add conv with filters 2 and relu activation at the end
		# Step 7: Add all the skip connections



		conv11 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu', padding ='same')(inputs)

		print("conv11")
		print(conv11)
		conv12 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu',  padding ='same')(conv11)
		
		print("conv12")
		print(conv12)

		#dont do padding
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(conv12)
		x = BatchNormalization()(x)

		print('x')
		print(x)

		conv21 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu',  padding ='same')(x)
		print("conv21")
		print(conv21)


		conv22 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu', padding ='same')(conv21)

		print("conv22")
		print(conv22)


		#dont do padding
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(conv22)
		x = BatchNormalization()(x)


		print("x-2")
		print(x)



		last1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu',  padding ='same')(x)

		print("last1")
		print(last1)
		last2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu',  padding ='same')(last1)

		print("last2")
		print(last2)

		#dont do padding
		#x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

		#x = BatchNormalization()(x)

		#
		x = UpSampling2D(size=(2,2))(last2)

		#x = Activation('relu')(x)
		print("adding dimentions")
		print(x)
		print(conv22)

		x = x + conv22
		deconv21 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu')(x)

		deconv22 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu')(deconv21)

		x = BatchNormalization()(deconv22)


		#add the unet connection
		x = UpSampling2D(size=(2,2))(x)
		#x = Activation('relu')(x)

		x = x + conv12
		deconv11 =Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu')(x)

		deconv12 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu')(deconv11)

		x = BatchNormalization()(deconv12)

		#output layer

		#x = UpSampling2D(input_shape=(128, 128, 16))(x)
		#x = Activation('relu')(x)



		x = Conv2D(2, kernel_size=(1, 1), strides=(1, 1),
		         activation='relu')(x)



		model = Model(input=inputs, output=x)

		model.summary()

		return model		
