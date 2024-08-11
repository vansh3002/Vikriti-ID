# models/generator.py
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Activation, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from .discriminator import unet_block 

def define_generator(image_shape):
    init = HeNormal()
    in_image = Input(shape=image_shape)

    # Contracting path
    s1, g1 = unet_block(64, in_image)
    p1 = MaxPooling2D(pool_size=(2, 2))(g1)
    s2, g2 = unet_block(128, p1)
    p2 = MaxPooling2D(pool_size=(2, 2))(g2)
    s3, g3 = unet_block(256, p2)
    p3 = MaxPooling2D(pool_size=(2, 2))(g3)
    s4, g4 = unet_block(512, p3)
    p4 = MaxPooling2D(pool_size=(2, 2))(g4)
    ## Bottleneck

    # Expansive path
    u1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(p4)
    c1 = Concatenate()([u1, g4, s4])
    _, g5 = unet_block(512, c1, add_skip=False)
    u2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(g5)
    c2 = Concatenate()([u2, g3, s3])
    _, g6 = unet_block(256, c2, add_skip=False)
    u3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(g6)
    c3 = Concatenate()([u3, g2, s2])
    _, g7 = unet_block(128, c3, add_skip=False)
    u4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(g7)
    c4 = Concatenate()([u4, g1, s1])
    _, g8 = unet_block(64, c4, add_skip=False)

    # Output layer
    g = Conv2D(1, (1, 1), padding='same', kernel_initializer=init)(g8)
    out_image = Activation('tanh')(g)

    # define model
    model = Model(in_image, out_image)
    return model
