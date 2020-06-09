from keras import initializers
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Dropout, Concatenate, Cropping2D, Conv2DTranspose
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
        init = initializers.RandomNormal(stddev=0.02)

        conv1 = Conv2D(64,
                        kernel_size=(5, 5),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(LeakyReLU(0.2))(conv1)

        conv11 = Conv2D(128,
                        kernel_size=(5, 5),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(conv1)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation(LeakyReLU(0.2))(conv11)

        conv21 = Conv2D(256,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(conv11)
        conv21 = BatchNormalization()(conv21)
        conv21 = Activation(LeakyReLU(0.2))(conv21)

        conv31 = Conv2D(512,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(conv21)
        conv31 = BatchNormalization()(conv31)
        conv31 = Activation(LeakyReLU(0.2))(conv31)

        # Bottleneck block
        bottleneck = conv31

        bnconv1 = Conv2D(512,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(bottleneck)
        bnconv1 = BatchNormalization()(bnconv1)
        bnconv1 = Activation(LeakyReLU(0.2))(bnconv1)

        convtrans31 = Conv2DTranspose(512,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(bnconv1)

        merge1 = Concatenate()([convtrans31, conv31])
        merge1 = Dropout(0.3)(merge1)

        deconv31 = Conv2D(512,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(merge1)
        deconv31 = BatchNormalization()(deconv31)
        deconv31 = Dropout(0.3)(deconv31)
        deconv31 = Activation(LeakyReLU(0.2))(deconv31)

        convtrans21 = Conv2DTranspose(256,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(deconv31)

        merge2 = Concatenate()([convtrans21, conv21])
        merge2 = Dropout(0.3)(merge2)

        deconv21 = Conv2D(256,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(merge2)
        deconv21 = BatchNormalization()(deconv21)
        deconv21 = Dropout(0.3)(deconv21)
        deconv21 = Activation(LeakyReLU(0.2))(deconv21)

        convtrans11 = Conv2DTranspose(128,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(deconv21)

        merge3 = Concatenate()([convtrans11, conv11])
        merge3 = Dropout(0.3)(merge3)

        deconv11 = Conv2D(128,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(merge3)
        deconv11 = BatchNormalization()(deconv11)
        deconv11 = Dropout(0.3)(deconv11)
        deconv11 = Activation(LeakyReLU(0.2))(deconv11)

        convtrans1 = Conv2DTranspose(64,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(deconv11)

        merge4 = Concatenate()([convtrans1, conv1])
        merge4 = Dropout(0.3)(merge4)

        deconv1 = Conv2D(64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(merge4)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Dropout(0.3)(deconv1)
        deconv1 = Activation(LeakyReLU(0.1))(deconv1)

        output = Conv2D(2,
                        kernel_size=(5, 5),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same',
                        activation='tanh')(deconv1)

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
        init = initializers.RandomNormal(stddev=0.02)

        conv11 = Conv2D(64,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(inputs)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation(LeakyReLU(0.2))(conv11)

        conv21 = Conv2D(128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(conv11)
        conv21 = BatchNormalization()(conv21)
        conv21 = Activation(LeakyReLU(0.2))(conv21)

        conv31 = Conv2D(256,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(conv21)
        conv31 = BatchNormalization()(conv31)
        conv31 = Activation(LeakyReLU(0.2))(conv31)
        conv31 = Dropout(0.2)(conv31)

        conv41 = Conv2D(512,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(conv31)
        conv41 = BatchNormalization()(conv41)
        conv41 = Activation(LeakyReLU(0.2))(conv41)
        conv41 = Dropout(0.2)(conv41)

        conv51 = Conv2D(1024,
                        kernel_size=(4, 4),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=init,
                        padding='same')(conv41)
        conv51 = BatchNormalization()(conv51)
        conv51 = Activation(LeakyReLU(0.2))(conv51)
        conv51 = Dropout(0.4)(conv51)

        output = Flatten()(conv51)
        output = Dense(1,
                       kernel_initializer=init,
                       use_bias=False)(output)
        output = BatchNormalization()(output)
        output = Activation('sigmoid')(output)

        model = Model(input=inputs, output=output)

        model.summary()

        return model	
