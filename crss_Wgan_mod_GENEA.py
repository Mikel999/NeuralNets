#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from keras.layers import Input, Conv1D, TimeDistributed, UpSampling1D, LSTM, SimpleRNN, GRU, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D####
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pickle
from functools import partial
from keras.layers.merge import _Merge
import tensorflow as tf
#K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
#np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 10000
n_critic=5
clip_value=0.01
cols=501#######14
rows=33
#######rows, cols = 4, 14
#input_shape = (4,14)
input_minmax_scaler = 0
#input_dim = cols * rows


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1))#original (32, 1, 1, 1)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_shape = (33,501)#original56
        self.latent_dim = 10000

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        #real_img = Input(shape=(self.img_shape,))
        real_img = Input(shape=(33,501))


        # Noise input
        #z_disc = Input(shape=(self.latent_dim,))
        z_disc = Input(shape=(33,self.latent_dim))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)
        print("AAAAAAAAAAAAAAAA",type(fake_img))
        print("AAAAAAAAAAAAAAAA",fake_img.shape)
        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        print(interpolated_img.shape,"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADADADAD")
        print(fake.shape,"fak")
        print(valid.shape,"val")
        # Determine validity of weighted sample
        #interpolated_img=tf.reshape(interpolated_img, shape=[32,4,60])
        print(interpolated_img.shape,"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADADADAD")
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(33,self.latent_dim))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(LSTM(501, return_sequences=True, unroll=True, input_shape=(33,self.latent_dim)))
        model.add(LSTM(501, return_sequences=True, unroll=True))
        
        #model.add(Conv1D(240,4,activation='tanh'))
        #model.summary()
        #model.add(Reshape((33,501)))
        #model.add(Dense(60,activation='tanh'))        
        
        
        #model.add(Conv1D(960,4,activation='relu',input_shape=(4,self.latent_dim)))
        #model.add(Reshape((4,240)))
        #model.add(MaxPooling1D(3))
        #model.add(Reshape((4,60)))
        #model.add(LSTM(100, return_sequences=True, unroll=True))
        #model.add(LSTM(60, return_sequences=True, unroll=True))
        
        model.summary()
        
        #noise = Input(shape=(self.latent_dim,))
        noise = Input(shape=(33,self.latent_dim))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()
        
        #model.add(Dense(512, input_shape=(240,), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        model.add(LSTM(501, return_sequences=True, unroll=True, input_shape=(33,501)))
        model.add(LSTM(501, return_sequences=True, unroll=True))
        #model.summary()
        #model.add(Conv1D(240,4,activation='relu'))
        #model.add(SimpleRNN(240, unroll=True))
        model.add(Reshape((16533,)))
        model.add(Dense(1,activation='linear'))
    
        model.summary()

        #img = Input(shape=(self.img_shape,))#original
        img = Input(shape=(33,501))
        validity = model(img)

        return Model(img, validity)






def myprint(model,dirtopology,model_name):
    with open(dirtopology+str(model_name)+'_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def sci_minmax(X,input_minmax_scaler=None):
    if input_minmax_scaler is None:
        input_minmax_scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
    return (input_minmax_scaler.fit_transform(X), input_minmax_scaler)


def import_data(filename=["../data/newmergeofData_025.txt","../data/newmergeofData_01.txt" ]):#folder,namefile):
    #mergeofData  input_training newmergeofData_025.txt newmergeofData_01.txt
    #input_data = np.loadtxt("../data/newmergeofData_025.txt",delimiter=';')
    print("Filename: ", filename[0])
    input_data =np.load(str(filename[0]))####
    #for i, f in enumerate(filename, start=1):
    #    input_data_temp = np.loadtxt(f, delimiter=';')
    #    #input_data2 = np.loadtxt("../data/newmergeofData_01.txt",delimiter=';')
    #    input_data=np.concatenate((input_data,input_data_temp),axis=0)

    (input_data, input_minmax_scaler) = sci_minmax(input_data)
    print("input dim ", input_data.shape)
    if (input_data.shape[0]%rows == 0):
        X_train = input_data.reshape( input_data.shape[0]//rows, cols* rows)
        print ("X_train dimension " , X_train.shape)
    else:
        input_data = input_data[:-(input_data.shape[0]%rows)]
        X_train = input_data.reshape( input_data.shape[0]//rows, cols* rows)
        print ("X_train dimension " , X_train.shape)

    return X_train,input_minmax_scaler

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

# Plot the loss from each batch
def plotLoss(epoch,dirr,batch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0,epoch])
    #plt.ylim([-1,2])
    plt.legend()
    plt.savefig(dirr+'/gan_loss_bs_'+str(batch)+'_epoch %d.png' % epoch)
    np.savetxt(dirr+'/bs_'+str(batch)+'_dLoss.csv', np.array(dLosses),fmt="%f",delimiter=',')
    np.savetxt(dirr+'/bs_'+str(batch)+'_gLoss.csv', np.array(gLosses),fmt="%f",delimiter=',')

#Save The generated Movements
def saveGeneratedMovements(epoch, dirr,batch,examples=100):
    noise = np.random.uniform(-1,1,size=[examples,33, randomDim])#CHANGE IF INPUT 1D, [examples, randomDim]
    generated_movements = wgan.generator.predict(noise)
    generated_movements = generated_movements.reshape(examples, rows, cols)
    print("MOVEMENEETSTESTSTSTSTS SHAPE", generated_movements.shape)
    for i,movement in enumerate(generated_movements, start=0):
        np.savetxt(dirr+"/n_batch_"+str(batch)+"_generatedMovements_epoch_"+str(epoch)+"_movement_"+str(i)+".csv", input_minmax_scaler.inverse_transform(movement),fmt="%f",delimiter=',')#original
        #np.savetxt(dirr+"/n_batch_"+str(batch)+"_generatedMovements_epoch_"+str(epoch)+"_movement_"+str(i)+".csv", movement,fmt="%f",delimiter=',')
    return

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch,dirmodels,gen,discr):
    gen.save(dirmodels + '/gan_generator_epoch_%d.h5' % epoch)
    discr.save(dirmodels + '/gan_discriminator_epoch_%d.h5' % epoch)


def loadModels(epoch,dirmodels):
    generatorLoad = load_model(dirmodels + '/gan_generator_epoch_%d.h5' % epoch)
    discriminatorLoad = load_model(dirmodels + '/gan_discriminator_epoch_%d.h5' % epoch)
    return generatorLoad, discriminatorLoad

def train(epochs=1, batchSize=32, unit_of_movement=4, n_example=100, train_file='/home/bee/robotak/rsait-crss/python/gan/generation/data/input_training.txt', mocap='openpose'):
    # Load the dataset
    global dLosses,gLosses
    dLosses = []
    gLosses = []
    global input_minmax_scaler
    X_train,input_minmax_scaler = import_data([train_file])
    batchCount = X_train.shape[0] / batchSize
    print ('Epochs:', epochs)
    print ('Batch size:', batchSize)
    print ('Batches per epoch:', batchCount)
    print(X_train.shape)
    X_train=X_train.reshape(-1,33,501)
    print(X_train.shape)
    # Adversarial ground truths
    valid = -np.ones((batchSize,1))
    fake =  np.ones((batchSize,1))
    dummy = np.zeros((batchSize,1)) # Dummy gt for gradient penalty
    print("val",valid.shape)
    
    for epoch in range(epochs+1):

        for _ in range(wgan.n_critic):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batchSize)
            imgs = X_train[idx]
            # Sample generator input
            noise = np.random.normal(0, 1, (batchSize,33,wgan.latent_dim))
            #print("Noise",noise.shape)

            # Train the critic
            d_loss = wgan.critic_model.train_on_batch([imgs, noise],
                                                            [valid, fake, dummy])#[valid, fake,dummy])

        # ---------------------
        #  Train Generator
        # ---------------------
        #for _ in range(wgan.n_critic):
        g_loss = wgan.generator_model.train_on_batch(noise, valid)
        # Plot the progress
        print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
        dLosses.append(d_loss)
        gLosses.append(g_loss)

        if epoch == 1:
            if(epoch==1):
                now = datetime.datetime.now()
                #timestamp=now.strftime("%Y-%m-%d-%H:%M")
                timestamp=now.strftime("%Y-%m-%d")
                dirr = '../samples/' + mocap + '/' + timestamp + '/' + mocap + '_n_epochs_' + str(epochs) + "_batchSize_" + str(batchSize) + "_um_" + str(unit_of_movement)# + '_' + str(timestamp)
                dirmodels = '../models/' + mocap + '/' + timestamp + '/' + mocap + '_n_epochs_' + str(epochs) + "_batchSize_" + str(batchSize) + "_um_" + str(unit_of_movement)# + '_' + str(timestamp)
                dirrloss = dirmodels + '/loss_file/'
                dirtopology = dirmodels + '/topology_net/'

        if(epoch == epochs):
            if not os.path.exists(dirr):
                os.makedirs(dirr)
            if not os.path.exists(dirrloss):
                os.makedirs(dirrloss)
            if not os.path.exists(dirmodels):
                os.makedirs(dirmodels)
            if not os.path.exists(dirtopology):
                os.makedirs(dirtopology)
            saveGeneratedMovements(epoch,dirr,batchSize,n_example)

    #save in a file the models
    myprint(wgan.critic,dirtopology,'discriminator')
    myprint(wgan.generator,dirtopology,'generator')
    #myprint(gan,dirtopology,'gan')


    plotLoss(epoch,dirrloss,batchSize) # Plot losses from every epoch
    saveModels(epoch, dirmodels, wgan.generator, wgan.critic)     #save a model
    filehandler = open(dirmodels+'/pickle_min_maxScaler', 'wb')  #save a pickle for min_max_scaler
    pickle.dump(input_minmax_scaler, filehandler, protocol=2)

if __name__ == '__main__':
    global rows, cols, input_dim
    parser = argparse.ArgumentParser()
    parser.add_argument("--e",type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--bs", type=int, default=32, help="Batch Size ")
    parser.add_argument("--um", type=int, default=4, help="Number of poses that composes a Unit of Movement")
    parser.add_argument("--ne", type=int, default=100, help="Number of example generated for each epochs")
    parser.add_argument("--db", type=str, default='/home/bee/robotak/rsait-crss/python/gan/generation/data/input_training.txt', help="Path of the training database")
    parser.add_argument("--mocap", type=str, default='openpose', help="Used motion capturing system")
    args = parser.parse_args()
    rows, cols = 33, 501#14
    input_dim = cols * rows
    wgan=WGANGP()
    train(args.e, args.bs, args.um, args.ne, args.db, args.mocap)
