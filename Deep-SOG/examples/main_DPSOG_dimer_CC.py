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
from sklearn.linear_model import RidgeCV
from SOG_Net.utilities import find_and_sort_neighbors_dimer
from SOG_Net.train import train_dimer
from SOG_Net.Neighbor import gen_coor_3d_species_three_body_dimer
from SOG_Net.DPSOG_dimer import DPSOG_dimer

print(tf.__version__)
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Set the number of threads
if tf.test.is_built_with_cuda():
  print("TensorFlow is compiled with GPU acceleration support")
else:
  print("TensorFlow is not compiled with GPU acceleration support")
tf.keras.backend.set_floatx('float32')

# Check if TensorFlow is currently running on a GPU
if tf.test.gpu_device_name():
  print("TensorFlow is currently using GPU:", tf.test.gpu_device_name())
else:
  print("TensorFlow is not running on a GPU")
gpus = tf.config.experimental.list_physical_devices('GPU')

### Get timestamp ###
now = datetime.datetime.now()
time_name0=now.strftime("%Y%m%d_%H%M%S")

# Input Data
# here we assume the data is generated within some cells. The number of cells in
# each dimension is "Ncells". "Np" shows the number of particles in per cell. 
# For simiplicity, we assume they are generated randomly uniformly.   
Nsamples = 12                          # 100 number of samples 
descriptorNet = [20, 40, 80]        # [2, 4, 8, 16, 32] size of descriptor network       
fittingNet = [64, 64, 64]    # [32, 32, 32, 32, 32, 32] size of fitting network
epochsPerStair = 1000                      # 10 decay step of learning rate   
learningRate = 0.001                    # 0.001 initial learning rate
decayRate = 0.995                         # 0.95 decay rate of learning rate
Nepochs = [40000, 8000, 8000] #[20000, 400, 400]         # [200, 400, 800, 1600] epoch
batchSizeArray = [1,2,4]#[4, 8, 12]            # [8, 16, 32, 64] batchsize      
maxNumNeighs = 23                      # Maximum number of neighbors
radious = 10                           # Short-range interaction radius
NpointsFourier = 31                    # Number of Fourier modes
fftChannels = 1                        # Number of FFT channels
DataType = "Periodic"                  # Data type
L = 30                                 # Total box length
xLims = [0.0, L]                       # Range in the x-direction
Npoints = 23                           # Total number of particles per configuration
Test_type = "SLR2"                     # Test type (e.g., SR, LR, SLR, SLR2)

print("Test_type",Test_type)
model_load = "Sum-of-Gaussian_Neural_Network/model_and_loss/Dimer/best_model/CC_SR.h5"

# read data file
dataFile="Sum-of-Gaussian_Neural_Network/dataset/dimer0_CC_data.h5";
#print(dataFile)

nameScript=Test_type+"_Nsamples_" + str(Nsamples) +  "_NpointsFourier_" + str(NpointsFourier) + "_radious_" + str(radious) + "_decayRate_" + str(decayRate) + "_"

#Folder for saving loss, accuracy and model
saveFolder  = "Sum-of-Gaussian_Neural_Network/model_and_loss/Dimer/CC_"

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsArray_Total = hf['points'][:,:Npoints]  # Array of particle coordinates
forcesArray_Total = hf['forces'][:,:Npoints]  # Array of forces
energyArray_Total = hf['energy'][:]  # Array of energies
energyArray_Total = np.squeeze(energyArray_Total, axis=-1)
chargesArray_Total = hf['charges'][:,:Npoints]  # Array of charges
chargesArray_Total = np.squeeze(chargesArray_Total)
pointsArray_Total[pointsArray_Total < 0.0] += L
pointsArray_Total[pointsArray_Total >= L] -= L

index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
pointsArray = pointsArray_Total[index]
forcesArray = forcesArray_Total[index]
energyArray = energyArray_Total[index]
chargesArray = chargesArray_Total[index]
chargesArray = np.squeeze(chargesArray)

pointsArray = tf.cast(pointsArray, tf.float32)
energyArray = tf.cast(energyArray, tf.float32)
forcesArray = tf.cast(forcesArray, tf.float32)
chargesArray = tf.cast(chargesArray, tf.float32)

###################   Pre-Fitting   ##################

X = np.zeros((Nsamples, 4), dtype=np.float32)
for i, val in enumerate([1, 2, 3, 4]):
    X[:, i] = np.sum(chargesArray == val, axis=1) 
Y = np.array(energyArray, dtype=np.float32) 
rcv = RidgeCV(alphas=np.geomspace(1e-8, 1e2, 10), fit_intercept=False)
rcv.fit(X, Y)
yp_base = rcv.predict(X)
y_diff = Y - yp_base
energyArray = y_diff.astype(np.float32)

#################    Pre-fitting completed   ################

Rinput = tf.Variable(pointsArray_Total, name="input", dtype=tf.float32)  # Input the array information into TensorFlow, allowing adjustments through backpropagation and optimization algorithms. pointsArray provides the static data for initialization, and the "name" parameter helps identify and manage this variable in the model.
Cinput = tf.Variable(chargesArray_Total, name="input", dtype=tf.float32)

# We only consider the first 100 configurations
Rin = Rinput[:2, :, :]  # Extract the first 100 configurations and corresponding elements from Rinput and assign them to the Rin array.
Rinnumpy = Rin.numpy()  # Convert the tensor Rin values into the Rinnumpy array for efficient analysis using NumPy operations.
Cin = Cinput[:2, :]  # Extract the first 100 configurations and corresponding elements from Cinput and assign them to the Cin array.
Cinnumpy = Cin.numpy()  # Convert the tensor Cin values into the Cinnumpy array for efficient analysis using NumPy operations.

Idx = find_and_sort_neighbors_dimer(Rinnumpy, Cinnumpy, L, radious, maxNumNeighs)
# Compute the neighbor list. Shape: (Nsamples, Npoints, MaxNumNeighs)
neighList = tf.Variable(Idx)  # Pass the generated neighbor list into the variable
genCoordinates, SRIJ = gen_coor_3d_species_three_body_dimer(Rin, neighList, L, 3 * radious / 4, radious)

# Compute the generated coordinates
filter = tf.cast(tf.reduce_sum(tf.abs(genCoordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = [0,1,2]).numpy()
numTotal = genCoordinates.shape[0] * genCoordinates.shape[1] * genCoordinates.shape[2]

# Mean is 0, variance is 1
av = tf.reduce_sum(genCoordinates, axis = [0,1,2], keepdims =True).numpy()[0]/numNonZero
std = np.sqrt((tf.reduce_sum(tf.square(genCoordinates - av), axis = [0,1,2], keepdims=True).numpy()[0] - av**2*(numTotal-numNonZero)) /numNonZero)
av = np.squeeze(av)
std = np.squeeze(std)

avTF = tf.constant(av, dtype=tf.float32)
stdTF = tf.constant(std, dtype=tf.float32)
## Define the model
model = DPSOG_dimer(Npoints, L, maxNumNeighs, descriptorNet, fittingNet, avTF, stdTF, NpointsFourier, fftChannels, xLims)

before_loss_time = time.time()
    
Rin2 = Rinput[:2,:,:] 
Cha2 = Cinput[:2,:]
Idx = find_and_sort_neighbors_dimer(Rin2.numpy(), Cha2.numpy(), L, radious, maxNumNeighs)

neigh_list2 = tf.Variable(Idx)

E, F = model(Rin2, Cha2, neigh_list2, radious, Test_type)
print(Rin2.dtype)

model.summary()

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

errorlist_energy = []
errorlist_force = []
errorlist_energy_train = []
errorlist_force_train = []
max_errorlist = []
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

# extracting the data
hf = h5py.File(dataFile, 'r')
arr = list(range(13))
index_to_remove = index
index_test = [arr[i] for i in range(len(arr)) if i not in index_to_remove]
pointsTest = hf['points'][index_test, :Npoints]  # Array of particle coordinates
forcesTest = hf['forces'][index_test, :Npoints]  # Array of forces
energyTest = hf['energy'][index_test]  # Array of energies
energyTest = np.squeeze(energyTest, axis=-1)
chargesTest = hf['charges'][index_test, :Npoints]  # Array of charges  
chargesTest = np.squeeze(chargesTest, axis=-1)
pointsTest[pointsTest < 0] += L
pointsTest[pointsTest >= L] -= L

####################################   Second fitting of energy   ###################################

X = np.zeros((len(index_test), 4), dtype=np.float32)
#print(X.shape)
for i, val in enumerate([1, 2, 3, 4]):
    X[:, i] = np.sum(chargesTest == val, axis=1) 
Y = np.array(energyTest, dtype=np.float32) 
print(X,chargesTest)
yp_base = rcv.predict(X)
y_diff = Y - yp_base
energyTest = y_diff.astype(np.float32)
print(energyTest)
print(yp_base)

######################################################################################

IdxTest = find_and_sort_neighbors_dimer(pointsTest, chargesTest, L, radious, maxNumNeighs)

neighListTest = tf.Variable(IdxTest)

#############################################  load model  ########################################
if Test_type == "SLR2":
  print("load Short model!")
  model.load_weights(model_load)
  model.compile()
  for layer in model.layers:
    print(f"{layer.name} is trainable: {layer.trainable}")
for layer in model.layers:
    new_weights = []
    for weight in layer.get_weights():
        # Only change the data type, keeping the shape unchanged
        new_weight = tf.cast(weight, tf.float32)
        new_weights.append(new_weight)
    
    # Set the converted weights
    layer.set_weights(new_weights)
print("Change double weights for load model")
for weight in model.trainable_weights:
    print(f"Weight: {weight.name}, dtype: {weight.dtype}")

###################training loop ##################################
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")
min_test_F_err = np.Inf

for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = 1.0
  weightF = 1000.0

  x_train = (pointsArray, energyArray, forcesArray, chargesArray)
  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL)
  
  # Iterate over epochs
  for epoch in range(epochs):
  
    loss_metric.reset_state()
    
    # Iterate over the batches of the dataset
    for step, x_batch_train in enumerate(train_dataset):
      start_epoch_time = time.time()

      Rinnumpy = x_batch_train[0].numpy()
      
      Idx = find_and_sort_neighbors_dimer(Rinnumpy, x_batch_train[3].numpy(), L, radious, maxNumNeighs)

      neighList = tf.Variable(Idx)
      
      loss, gradients = train_dimer(model, optimizer, mse_loss_fn, x_batch_train[0], neighList, x_batch_train[3], radious, x_batch_train[1], x_batch_train[2], weightE, weightF, Test_type)
      loss_metric(loss)
      
      '''
      if step % 10 == 0:
        print(f'Train outloss took {after_loss_time - before_loss_time:.6f} seconds')

      if step % 10 == 0:
        print('step %s: total loss = %s' % (step, str(loss.numpy())))
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))
      #end_epoch_time = time.time()
      #print(f'Epoch took {end_epoch_time - start_epoch_time:.6f} seconds')
      '''

    r_input = pointsArray[:,:,:]
    Idx = find_and_sort_neighbors_dimer(r_input.numpy(), chargesArray[:,:].numpy(), L, radious, maxNumNeighs)
    neighList = tf.Variable(Idx)
    pottrain, forcetrain = model(r_input, chargesArray[:,:], neighList, radious, Test_type)
    err_train = tf.sqrt(tf.reduce_mean(tf.square(forcetrain - forcesArray[:,:,:])))

    '''
    #print(f"That's ok,2")
    neighList_r_input = tf.Variable(Idx_r_input)
    pottrain, forcetrain = model(r_input, chargesArray, neighList_r_input, radious, Test_type)
    
    #print(forcetrain)
    #print(f"That's begin")
    err_train = tf.sqrt(tf.reduce_mean(tf.square(forcetrain - forcesArray[:,:,:].numpy())))
    #print(f"That's ok,1")
    #print(forcesArray[1,:,:])
    #AA = forcetrain - forcesArray[:,:,:]
    #print(forcetrain.shape, forcesArray.shape, AA.shape)
    '''
   
    err_ener_train = tf.sqrt(tf.reduce_mean(tf.square(pottrain - energyArray[:])))#/tf.sqrt(tf.reduce_sum(tf.square(pottrain)))
  
    potPred, forcePred = model(pointsTest, chargesTest, neighListTest, radious, Test_type)

    err = tf.sqrt(tf.reduce_mean(tf.square(forcePred - forcesTest)))

    err_ener = tf.sqrt(tf.reduce_mean(tf.square(potPred - energyTest)))#/tf.sqrt(tf.reduce_sum(tf.square(potPred)))

    if err_ener_train<2e-2 and err_ener<2e-2:
      model.save_weights(saveFolder+time_name0+'_tol2.h5', save_format='h5')
      if err_ener_train<1e-2 and err_ener<2e-2:
        model.save_weights(saveFolder+time_name0+'_tol12.h5', save_format='h5')
      if err_ener_train<2e-2 and err_ener<1e-2:
        model.save_weights(saveFolder+time_name0+'_tol21.h5', save_format='h5')
      print("Relative Error in the trained forces is " +str(err_train.numpy()))
      print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
      print("Relative Error in the test forces is " +str(err.numpy()))
      print("Relative Error in the test energy is " +str(err_ener.numpy()))
    
    # save the error
    errorlist_force.append(err.numpy()) 
    with open(saveFolder+time_name0+'_error_force_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force)
    errorlist_energy.append(err_ener.numpy())
    with open(saveFolder+time_name0+'_error_energy_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy) 
    errorlist_force_train.append(err_train.numpy())
    with open(saveFolder+time_name0+'_error_force_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force_train)
    errorlist_energy_train.append(err_ener_train.numpy()) 
    with open(saveFolder+time_name0+'_error_energy_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy_train) 
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # decay learning rate 
    
    lr_value = optimizer.learning_rate(optimizer.iterations).numpy()#(optimizer.learning_rate).numpy()
    lrStr = str(lr_value)
    
    if epoch%100 ==0:
      print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,meanLossStr,lrStr))
    
    # save the loss
    losslist.append(loss_metric.result().numpy())
    with open(saveFolder+time_name0+'_loss_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(losslist)
    
  print("saving the weights")
  ###  Get timestamp  ###
  str_num = str(cycle)

  model.save_weights(saveFolder+time_name+'my_model_'+str_num+'.h5', save_format='h5')



