from keras.layers import Input
from keras.models import Model


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

		model = Model(input=inputs, output=inputs)

		model.summary()

		return model		
