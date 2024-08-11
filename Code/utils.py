import numpy as np
from numpy import load
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import img_to_array
from numpy.random import randint
import matplotlib.pyplot as pyplot

def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['dataA'], data['dataB']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape,npatch, n_samplesy):
    ix = randint(0, dataset.shape[0],n_samples) 
    X = dataset[ix]
    y = np.ones((n_samplesy, patch_shape, npatch, 1))
    return X, y

def generate_fake_samples(g_model, dataset, patch_shape,npatch):
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, npatch, 1))
    return X, y

def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif np.random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)

def summarize_performance(step, g_model, trainX, name, n_samples=5,n_samplesy=5):
   X_in, _ = generate_real_samples(trainX, n_samples, 0 ,0,n_samplesy)
   X_out, _ = generate_fake_samples(g_model, X_in, 0,0)
   X_in = (X_in + 1) / 2.0
   X_out = (X_out + 1) / 2.0
   for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(X_in[i, :, :, 0], cmap='gray')
   for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(X_out[i, :, :, 0], cmap='gray')
   filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
   pyplot.savefig(filename1)
   pyplot.close()

def save_models(step, g_model_AtoB, g_model_BtoA):
   filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
   g_model_AtoB.save(filename1)
   filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
   g_model_BtoA.save(filename2)
   print('>Saved: %s and %s' % (filename1, filename2))
