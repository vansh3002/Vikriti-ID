# models/discriminator.py
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def define_discriminator(image_shape):
    init = HeNormal()
    in_image = Input(shape=image_shape)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model

def unet_block(n_filters, input_layer, add_skip=True):
    init = HeNormal()
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init, kernel_regularizer=l2(0.001))(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = LeakyReLU(alpha=0.2)(g)
    skip = g if add_skip else None
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init, kernel_regularizer=l2(0.001))(g)
    g = InstanceNormalization(axis=-1)(g)
    g = LeakyReLU(alpha=0.2)(g)
    return skip, g
