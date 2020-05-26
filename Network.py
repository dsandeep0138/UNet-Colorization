from keras.models import Model
from keras.layers import Input, Reshape, Dense, Dropout, Concatenate, Cropping2D
from keras.layers import Activation, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.layers import Flatten
from keras.layers.advanced_activations import LeakyReLU
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


class GanGenerator(object):
    def __init__(self, num_layers, num_filters):
        self.num_layers = num_layers
        self.num_filters = num_filters

    def build_model(self):
        # Input to the model
        inputs = Input(shape=(256, 256, 1), name = 'image_input')

        conv11 = Conv2D(32,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(inputs)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation(LeakyReLU(0.2))(conv11)

        conv21 = Conv2D(64,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(conv11)
        conv21 = BatchNormalization()(conv21)
        conv21 = Activation(LeakyReLU(0.2))(conv21)

        conv31 = Conv2D(128,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(conv21)
        conv31 = BatchNormalization()(conv31)
        conv31 = Activation(LeakyReLU(0.2))(conv31)

        # Bottleneck block
        bottleneck = conv31

        bnconv1 = Conv2D(256,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(bottleneck)
        bnconv1 = BatchNormalization()(bnconv1)
        bnconv1 = Activation(LeakyReLU(0.2))(bnconv1)

        up1 = UpSampling2D(size=(2,2))(bnconv1)

        merge1 = Concatenate()([up1, conv31])

        deconv31 = Conv2D(128,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same')(merge1)
        deconv31 = BatchNormalization()(deconv31)
        deconv31 = Dropout(0.5)(deconv31)
        deconv31 = Activation('relu')(deconv31)

        up2 = UpSampling2D(size=(2,2))(deconv31)

        merge2 = Concatenate()([up2, conv21])
		
        deconv21 = Conv2D(64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same')(merge2)
        deconv21 = BatchNormalization()(deconv21)
        deconv21 = Dropout(0.5)(deconv21)
        deconv21 = Activation('relu')(deconv21)

        up3 = UpSampling2D(size=(2,2))(deconv21)

        merge3 = Concatenate()([up3, conv11])

        deconv11 = Conv2D(32,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same')(merge3)
        deconv11 = BatchNormalization()(deconv11)
        deconv11 = Activation('relu')(deconv11)

        up4 = UpSampling2D(size=(2,2))(deconv11)

        output = Conv2D(2,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='tanh')(up4)

        model = Model(input=inputs, output=output)

        model.summary()
		
        return model	


class GanDiscriminator(object):

    def __init__(self, num_layers, num_filters):
        self.num_layers = num_layers
        self.num_filters = num_filters

    def build_model(self):
        # Input to the model
        inputs = Input(shape=(256, 256, 2), name = 'image_input')

        conv11 = Conv2D(32,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(inputs)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation(LeakyReLU(0.2))(conv11)

        conv21 = Conv2D(64,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(conv11)
        conv21 = BatchNormalization()(conv21)
        conv21 = Activation(LeakyReLU(0.2))(conv21)

        conv31 = Conv2D(128,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(conv21)
        conv31 = BatchNormalization()(conv31)
        conv31 = Activation(LeakyReLU(0.2))(conv31)

        conv41 = Conv2D(256,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same')(conv31)
        conv41 = BatchNormalization()(conv41)
        conv41 = Activation(LeakyReLU(0.2))(conv41)

        output = Flatten()(conv41)
        output = Dense(1)(output)
        output = Activation('sigmoid')(output)

        model = Model(input=inputs, output=output)

        model.summary()

        return model	
