# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import rotate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from numpy import savez_compressed


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


custom_objects = {'Sampling': Sampling}
encoder = load_model("C://Vikrit-Id//encoder.h5", custom_objects=custom_objects)
decoder = load_model("C://Vikrit-Id//decoder.h5", custom_objects=custom_objects)


encoder_input = encoder.input
encoder_output = encoder.layers[-1].output
decoder_output = decoder(encoder_output)
combined_model = tf.keras.models.Model(inputs=encoder_input, outputs=decoder_output)


def load_images(path, size=(400, 256, 1)):
    data_list = []
    for filename in os.listdir(path):
        pixels = Image.open(os.path.join(path, filename)).convert('L')
        pixels = pixels.resize((size[1], size[0]))
        pixels = np.array(pixels)
        pixels = np.expand_dims(pixels, axis=-1)
        data_list.append(pixels)
    return np.asarray(data_list)

def process_noise_and_save(path, save_path):
    dataB = []
    for i in range(100):  
        noise = np.random.uniform(size=(200, 136, 1)).astype(np.float32)
        noise = np.expand_dims(noise, axis=0)
        result = combined_model.predict(noise)
        result_resized = tf.image.resize(result, (400, 256))
        result_resized = tf.squeeze(result_resized, axis=0)
        dataB.append(result_resized)
    dataB = np.asarray(dataB)
    dataA = load_images(path)[:100]
    savez_compressed(save_path, dataA=dataB, dataB=dataA)
    print(f"Saved data to {save_path}")

# Paths to dataset directories
dataset_path = 'C://Vikrit-Id//fp_1'
save_path = 'C://Vikrit-Id//processed_data.npz'

# Process noise and save results
if __name__ == '__main__':
    process_noise_and_save(dataset_path, save_path)
