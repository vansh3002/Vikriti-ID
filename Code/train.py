from tensorflow.keras import backend as K
from utils import load_real_samples, generate_real_samples, generate_fake_samples, update_image_pool, summarize_performance, save_models
from models.generator import define_generator
from models.discriminator import define_discriminator
from models.composite_model import define_composite_model


def lr_schedule(epoch):
    initial_lr = 0.0002
    if epoch > 10000:
        lr = initial_lr * 1
    elif epoch > 4000:
        lr = initial_lr * 1
    elif epoch > 1000:
        lr = initial_lr * 1
    else:
        lr = initial_lr
    return lr

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset1):
    n_epochs, n_batch = 10, 1
    n_patch = d_model_A.output_shape[1]
    npatchy = d_model_A.output_shape[2]
    
    trainA, trainB = dataset1
    poolA, poolB = list(), list()
    bat_per_epo = int(min(len(trainA), len(trainB)) / n_batch)
    n_steps = bat_per_epo * n_epochs
    
    for i in range (500):
        n_batchy=1
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch,npatchy,n_batchy)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch,npatchy,n_batchy)
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch,npatchy)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch,npatchy)
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        lr = lr_schedule(i)
        K.set_value(c_model_AtoB.optimizer.learning_rate, lr)
        K.set_value(c_model_BtoA.optimizer.learning_rate, lr)
        K.set_value(d_model_A.optimizer.learning_rate, lr)
        K.set_value(d_model_B.optimizer.learning_rate, lr)
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
        epochs_per_summary = 60 
        epochs_per_save = 500
        if (((i+1) % 10 == 0) or i==0):
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
        if (((i+1) % 10 == 0) and i!=0):
            save_models(i, g_model_AtoB, g_model_BtoA)   
    return

if __name__ == '__main__':
    dataset1 = load_real_samples("C://Vikrit-Id//processed_data.npz")
    image_shape = dataset1[0].shape[1:]
    
    d_model_A = define_discriminator(image_shape)
    d_model_B = define_discriminator(image_shape)
    g_model_AtoB = define_generator(image_shape)
    g_model_BtoA = define_generator(image_shape)
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_A, g_model_BtoA, image_shape)
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_B, g_model_AtoB, image_shape)
    
    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset1)
