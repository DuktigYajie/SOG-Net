import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os.path
from os import path
import h5py
import sys
import json
import time
import csv
import datetime  # Timestamp

from SOG_Net.utilities import gen_coor_3d
from SOG_Net.train import train_pointcharge
from SOG_Net.DPSOG_pointcharge import DPSOG_pointcharge

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if tf.test.is_built_with_cuda():
  print("TensorFlow is compiled with GPU support")
else:
  print("TensorFlow is not compiled with GPU support")

# Check if TensorFlow is currently executing on a GPU
if tf.test.gpu_device_name():
  print("TensorFlow is currently using GPU:", tf.test.gpu_device_name())
else:
  print("TensorFlow is not executing on a GPU")
gpus = tf.config.experimental.list_physical_devices('GPU')

### Get timestamp ###
now = datetime.datetime.now()
time_name0=now.strftime("%Y%m%d_%H%M%S")

# Input Data
# here we assume the data is generated within some cells. The number of cells in
# Each dimension is "Ncells". "Np" shows the number of particles per cell. 
# For simplicity, we assume they are generated randomly and uniformly.   
Nsamples = 1000                        # Number of samples 
descriptorNet = [2, 4, 8, 16, 32]      # Size of descriptor network       
fittingNet = [32, 32, 32, 32, 32, 32]  # Size of fitting network
epochsPerStair = 100                   # Decay step of learning rate   
learningRate = 0.001                   # Initial learning rate
decayRate = 0.99                       # Decay rate of learning rate
Nepochs = [200, 400, 800, 1600]        # Epochs for training
batchSizeArray = [8, 16, 32, 64]       # Batch sizes for training      
maxNumNeighs = 20                      # Maximum number of neighbors
radious = 1.5                          # Short-range interaction radius 
NpointsFourier = 17                    # Number of Fourier modes 
fftChannels = 1                        # Number of FFT channels 
DataType = "Periodic"                  # Data type
L = 10                                 # Total box length
xLims = [0.0, L]                       # Range in the x-direction
Npoints = 200                          # Total number of particles per configuration

# read data file
dataFile="Sum-of-Gaussian_Neural_Network/dataset/pointcharge_data_train.h5";
print(dataFile)

nameScript="_Nsamples_" + str(Nsamples) +  "_NpointsFourier_" + str(NpointsFourier) + "_radious_" + str(radious) + "_decayRate_" + str(decayRate) 
# Folder for saving loss, accuracy, and model
saveFolder = "Sum-of-Gaussian_Neural_Network/model_and_loss/pointcharge/"

# Extracting the data
hf = h5py.File(dataFile, 'r')

pointsArray = hf['points'][:]  # Array of point coordinates
forcesArray = hf['forces'][:]  # Array of forces
energyArray = hf['energy'][:]  # Array of energies
chargesArray = hf['charges'][:]  # Array of charges

pointsArray[pointsArray < 0.0] += L
pointsArray[pointsArray >= L] -= L

pointsArray = np.transpose(pointsArray, axes=(2, 1, 0))
forcesArray = np.transpose(forcesArray, axes=(2, 1, 0))
energyArray = np.transpose(energyArray, axes=(1, 0))
chargesArray = np.transpose(chargesArray, axes=(1, 0))

Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32) # Input the array information into TensorFlow, allowing adjustments through backpropagation and optimization algorithms. pointsArray provides the static data for initialization, and the "name" parameter helps identify and manage this variable within the model.
Cinput = tf.Variable(chargesArray, name="input", dtype = tf.float32)

# we only consider the first 100 
Rin = Rinput[:100,:,:] # Extract the first 100 configurations and all corresponding elements from Rinput, assigning them to the Rin array
Rinnumpy = Rin.numpy() # Convert the tensor Rin into the Rinnumpy array to enable efficient analysis using numpy operations

#This is the new implementation
n_samples, n_points, dimension = Rinnumpy.shape
Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1

for i in range(n_samples):
    r_sample = Rinnumpy[i]
    tree = cKDTree(r_sample, boxsize=[L,L,L])
    r_list = tree.query_ball_point(r_sample,radious)
    r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]    
    for j,row in enumerate(r_list):
      Idx[i,j,:len(row)]=row

# compute the neighbor list. shape:(Nsamples, Npoints and MaxNumneighs)
neighList = tf.Variable(Idx) # Pass the generated neighbor list into the variable

genCoordinates = gen_coor_3d(Rin, neighList, L)
# compute the generated coordinates
filter = tf.cast(tf.reduce_sum(tf.abs(genCoordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = 0).numpy() # Number of non-zero elements: 7941106
numTotal = genCoordinates.shape[0] # Total number of elements: 12000000

# Mean is 0, variance is 1
av = tf.reduce_sum(genCoordinates, axis = 0, keepdims =True).numpy()[0]/numNonZero
std = np.sqrt((tf.reduce_sum(tf.square(genCoordinates - av), axis = 0, keepdims=True).numpy()[0] - av**2*(numTotal-numNonZero)) /numNonZero)

avTF = tf.constant(av, dtype=tf.float32)
stdTF = tf.constant(std, dtype=tf.float32)
## Define the model
model = DPSOG_pointcharge(Npoints, L, maxNumNeighs, descriptorNet, fittingNet, avTF, stdTF, NpointsFourier, fftChannels, xLims)

# quick run of the model to check that it is correct.
#This is the new implementation
Rin2 = Rinput[:2,:,:] 
Cha2 = Cinput[:2,:]
n_samples, n_points, dimension = Rin2.shape
Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
for i in range(n_samples):
    r_sample = Rinnumpy[i]
    tree = cKDTree(r_sample, boxsize=[L,L,L])
    r_list = tree.query_ball_point(r_sample,radious)
    r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]    
    for j,row in enumerate(r_list):
      Idx[i,j,:len(row)]=row

neigh_list2 = tf.Variable(Idx)

E, F = model(Rin2, Cha2, neigh_list2)

model.summary()

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

errorlist = []
losslist = []

### optimization parameters ##
mse_loss_fn = tf.keras.losses.MeanSquaredError()

initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps=(Nsamples//batchSizeArray[0])*epochsPerStair,
             decay_rate=decayRate,
             staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)

loss_metric = tf.keras.metrics.Mean()

# read data file of test set
dataFile="Sum-of-Gaussian_Neural_Network/dataset/pointcharge_data_test.h5";
print(dataFile)

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsTest = hf['points'][:]  # Array of point coordinates
forcesTest = hf['forces'][:]  # Array of forces
energyTest = hf['energy'][:]  # Array of energies
chargesTest = hf['charges'][:]  # Array of charges

pointsTest[pointsTest < 0] += L
pointsTest[pointsTest >= L] -= L

pointsTest = np.transpose(pointsTest, axes=(2, 1, 0))
forcesTest = np.transpose(forcesTest, axes=(2, 1, 0))
energyTest = np.transpose(energyTest, axes=(1, 0))
chargesTest = np.transpose(chargesTest, axes=(1, 0))

#This is the new implementation
n_samples, n_points, dimension = pointsTest.shape
IdxTest = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
for i in range(n_samples):
    r_sample = pointsTest[i]
    tree = cKDTree(r_sample, boxsize=[L,L,L])
    r_list = tree.query_ball_point(r_sample,radious)
    r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]    
    for j,row in enumerate(r_list):
      IdxTest[i,j,:len(row)]=row
neighListTest = tf.Variable(IdxTest)

###################training loop ##################################
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")

for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = 0.0
  weightF = 1.0

  x_train = (pointsArray, energyArray, forcesArray, chargesArray)

  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL)

  # Iterate over epochs
  for epoch in range(epochs):
    start = time.time()
    print('============================', flush = True) 
    print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_state()
    
    # Iterate over the batches of the dataset
    for step, x_batch_train in enumerate(train_dataset):
      start_epoch_time = time.time()
      
      #print(x_batch_train[0].shape)

      Rinnumpy = x_batch_train[0].numpy()

      #This is the old implementation
      #Idx = comput_inter_list(Rinnumpy, L, radious, maxNumNeighs)
      
      #This is the new implementation
      n_samples, n_points, dimension = Rinnumpy.shape
      Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
      for i in range(n_samples):
        r_sample=Rinnumpy[i]
        tree=cKDTree(r_sample,boxsize=[L,L,L])
        r_list = tree.query_ball_point(r_sample,radious)
        r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)] 
        for j,row in enumerate(r_list):
          Idx[i,j,:len(row)]=row

      neighList = tf.Variable(Idx)
      
      before_loss_time = time.time()

      # x_batch_train[0] input, x_batch_train[1] energy output, x_batch_train[2] force output
      loss,_ = train_pointcharge(model, optimizer, mse_loss_fn, x_batch_train[0], neighList, x_batch_train[3], x_batch_train[1], x_batch_train[2], weightE, weightF)
                           
      after_loss_time = time.time()
      

      loss_metric(loss)
      if step % 10 == 0:
        print('step %s: total loss = %s' % (step, str(loss.numpy())))
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))
      #end_epoch_time = time.time()
      #print(f'Epoch took {end_epoch_time - start_epoch_time:.6f} seconds')
    
    r_input = pointsArray[:10,:,:]
    n_samples, n_points, dimension = r_input.shape
    Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
    for i in range(n_samples):
      r_sample = r_input[i]
      tree = cKDTree(r_sample, boxsize=[L,L,L])
      r_list = tree.query_ball_point(r_sample,radious)
      r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]  
      for j,row in enumerate(r_list):
        Idx[i,j,:len(row)]=row
    neighList = tf.Variable(Idx)
    
    pottrain, forcetrain = model(r_input, chargesArray[:10,:], neighList)

    errtrain = tf.sqrt(tf.reduce_sum(tf.square(forcetrain - forcesArray[:10,:,:])))\
               /tf.sqrt(tf.reduce_sum(tf.square(forcetrain)))

    err_ener_train=tf.sqrt(tf.reduce_sum(tf.square(pottrain-energyArray[:10,:])))/tf.sqrt(tf.reduce_sum(tf.square(pottrain)))

    print("Relative Error in the trained forces is " +str(errtrain.numpy()))

    potPred, forcePred = model(pointsTest, chargesTest, neighListTest)
    print(forcePred.shape,forcesTest.shape)
    err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
    print(potPred.shape,energyTest.shape)
    err_ener=tf.sqrt(tf.reduce_sum(tf.square(potPred - energyTest)))/tf.sqrt(tf.reduce_sum(tf.square(potPred)))

    print("Relative Error in the forces is " +str(err.numpy()))

    end = time.time()
    print('time elapsed %.4f'%(end - start))
    
    # save the error
    errorlist.append(err.numpy())
    with open(saveFolder+time_name0+'_error_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist)
        
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # decay learning rate 
    
    #lrStr = str(optimizer._decayed_lr('float32').numpy())
    lr_value = optimizer.learning_rate(optimizer.iterations).numpy()#(optimizer.learning_rate).numpy()
    lrStr = str(lr_value)
    
    print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,
                                                          meanLossStr,
                                                          lrStr))
    
    # save the loss
    losslist.append(loss_metric.result().numpy())
    with open(saveFolder+time_name0+'_loss_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(losslist)
    
  print("saving the weights")
  ### Get timestamp ###
  str_num = str(cycle)

  model.save_weights(saveFolder+time_name+'my_model_'+str_num+'.h5', save_format='h5')



