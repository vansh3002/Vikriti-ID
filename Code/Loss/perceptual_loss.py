from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import MeanSquaredError

def perceptual_loss(y_true, y_pred):
    vgg_model = VGG19(weights='imagenet', include_top=False)
    perceptual_layers = ['block3_conv3', 'block4_conv3']
    perceptual_model = Model(inputs=vgg_model.input, outputs=[vgg_model.get_layer(layer).output for layer in perceptual_layers])
    y_true_features = perceptual_model(y_true)
    y_pred_features = perceptual_model(y_pred)
    mse_loss = MeanSquaredError()(y_true_features, y_pred_features)
    return mse_loss

