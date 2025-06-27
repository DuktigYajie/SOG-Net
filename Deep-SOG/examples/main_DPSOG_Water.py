import tensorflow as tf
import numpy as np
import os.path
from os import path
import h5py
import time
import csv
import datetime  # Timestamp
from SOG_Net.utilities import find_and_sort_neighbors_water
from SOG_Net.train import train_Water

from SOG_Net.DPSOG_Water import DPSOG_Water
#from tensorflow.keras import mixed_precision

import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Set the number of threads
if tf.test.is_built_with_cuda():
    print("TensorFlow is compiled with GPU acceleration support")
else:
    print("TensorFlow is not compiled with GPU acceleration support")
# Check if TensorFlow is currently executing on a GPU
if tf.test.gpu_device_name():
    print("TensorFlow is currently using GPU:", tf.test.gpu_device_name())
else:
    print("TensorFlow is not currently executing on a GPU")
gpus = tf.config.experimental.list_physical_devices('GPU')

#mixed_precision.set_global_policy('float32')

# Check the current global precision policy
#print("Current global precision policy:", mixed_precision.global_policy().name)

### Get timestamp ###
now = datetime.datetime.now()
time_name0 = now.strftime("%Y%m%d_%H%M%S")

################# Input Data ################
# Here we assume the data is generated within some cells. The number of cells in
# each dimension is "Ncells". "Np" shows the number of particles per cell. 
# For simplicity, we assume they are generated randomly and uniformly.   
Nsamples = 1000                          # Number of samples (e.g., 1000)
descriptorNet = [25, 50, 100]            # Size of descriptor network (e.g., [25, 50, 100])       
fittingNet = [120, 120, 120]             # Size of fitting network (e.g., [120, 120, 120])
epochsPerStair = 1000                    # Decay step of learning rate (e.g., 1000 epochs)   
learningRate = 0.001                     # Initial learning rate (e.g., 0.001)
decayRate = 0.99                         # Decay rate of learning rate (e.g., 0.99)
Nepochs = [40000, 40000]                 # Number of epochs per cycle (e.g., [40000, 40000])
batchSizeArray = [50, 100]               # Batch sizes for training (e.g., [50, 100])      
maxNumNeighs_O_A = 25                    # Maximum number of oxygen neighbors in the A region
maxNumNeighs_H_A = 50                    # Maximum number of hydrogen neighbors in the A region
maxNumNeighs_O_R = 60                    # Maximum number of oxygen neighbors in the R region
maxNumNeighs_H_R = 120                   # Maximum number of hydrogen neighbors in the R region
radious_A = 5                            # Short-range interaction radius (e.g., 5 units)
radious_R = 7 
NpointsFourier = 5                       # Number of Fourier modes
fftChannels = 1                          # Number of FFT channels
DataType = "Periodic"                    # Data type
L = 14.4088                              # Box length
Npoints = 300                            # Total number of particles per configuration
select_neuron = 16                       # Number of selected invariant features
Test_type = "SLR2"                       # Test type (e.g., SR, LR, SLR, SLR2)
Energy_rate = 0.0                        # Energy contribution rate
Force_rate = 1.0                         # Force contribution rate
Load_short = 0                           # Load/unload the existing model
inner_factor_A = 1.0 / 8.0               # Starting position for softening (A region)
inner_factor_R = 1.0 / 14.0              # Starting position for softening (R region)

model_load = "Sum-of-Gaussian_Neural_Network/dataset/pointcharge_data_test.h5"
mirrored_strategy = tf.distribute.MirroredStrategy()

# read data file
dataFile = "../dataset/water_1900_data.h5"
hf = h5py.File(dataFile, 'r')
pointsArray_Total = hf['points'][:]  # Array of point coordinates
forcesArray_Total = hf['forces'][:]  # Array of forces
energyArray_Total = hf['energy'][:]  # Array of energies
energyArray_Total = np.squeeze(energyArray_Total, axis=-1)  # Remove the last dimension if it is singleton
chargesArray_Total = hf['charges'][:]  # Array of charges
chargesArray_Total = np.squeeze(chargesArray_Total)
nameScript="_Nsamples_" + str(Nsamples) +  "_NpointsFourier_" + str(NpointsFourier) + "_radious_A_" + str(radious_A) + "_radious_R_" + str(radious_R) + "_decayRate_" + str(decayRate) + "_water_" + Test_type + "_energyRate_" + str(Energy_rate) + "_forceRate_" + str(Force_rate) + "_inner_factor_A_" + str(inner_factor_A) + "_inner_factor_R_" + str(inner_factor_R)
#Folder for saving loss, accuracy and model
saveFolder  = "model_and_loss/water/"

print(pointsArray_Total.shape)

# extracting the data: 90% of dataset
pointsArray = pointsArray_Total[:(Nsamples//10*9),:Npoints]
forcesArray = forcesArray_Total[:(Nsamples//10*9),:Npoints]
energyArray = energyArray_Total[:(Nsamples//10*9)]
chargesArray = chargesArray_Total[:(Nsamples//10*9),:Npoints]

# Input array information into TensorFlow, allowing adjustments through backpropagation and optimization algorithms. 
# pointsArray provides static data as initial values, and the "name" parameter helps identify and manage this variable within the model.
Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)
Cinput = tf.Variable(chargesArray, name="input", dtype = tf.float32) 

# read data file of test set: 10% of dataset
pointsTest = pointsArray_Total[(Nsamples//10*9):Nsamples,:Npoints]  # Position
forcesTest = forcesArray_Total[(Nsamples//10*9):Nsamples,:Npoints]  # Force
energyTest = energyArray_Total[(Nsamples//10*9):Nsamples]  # Energy
energyTest = np.squeeze(energyTest)
chargesTest = chargesArray_Total[(Nsamples//10*9):Nsamples,:Npoints]  # Charge  
chargesTest = np.squeeze(chargesTest)

## Define the model
model = DPSOG_Water(Npoints, maxNumNeighs_O_A, maxNumNeighs_H_A, maxNumNeighs_O_R, maxNumNeighs_H_R, L, descriptorNet, fittingNet, NpointsFourier, fftChannels)

## Test Model ###############################################################################################################################
Rin2 = pointsArray[:10] 
Cha2 = chargesArray[:10]
#before_loss_time = time.time()
Idx_O_A, Idx_H_A, Idx_O_R, Idx_H_R = find_and_sort_neighbors_water(Rin2, Cha2, L, radious_A, maxNumNeighs_O_A, maxNumNeighs_H_A, radious_R, maxNumNeighs_O_R, maxNumNeighs_H_R)
# Further distinguish the target for O and H
mask_Cha2_O = tf.equal(Cha2, 1)  # Create a boolean mask of shape (2, 300) for O
mask_Cha2_H = tf.equal(Cha2, 2)  # Create a boolean mask of shape (2, 300) for H
for i in ["O", "H"]: 
    for j in ["O", "H"]:
        for k in ["A", "R"]:
        exec(f"Idx_{i}_{j}_{k} = []")  # Initialize empty lists for indices
for m in range(Rin2.shape[0]):  # Iterate over the batch
    for i in ["O", "H"]:
    exec(f"batch_mask_{i} = mask_Cha2_{i}[m]")  # Current batch mask
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"selected_{i}_{j}_{k} = tf.boolean_mask(Idx_{j}_{k}[m], batch_mask_{i})")
                exec(f"Idx_{i}_{j}_{k}.append(selected_{i}_{j}_{k})")
for i in ["O", "H"]:
    for j in ["O", "H"]:
        for k in ["A", "R"]:
            exec(f"Idx_{i}_{j}_{k} =  tf.stack(Idx_{i}_{j}_{k}, axis=0)")
            exec(f"print(Idx_{i}_{j}_{k}.shape)")

LArray2 = tf.Variable(L, name="input", dtype = tf.float32)
E, F = model(Rin2, Cha2, tf.cast(LArray2, dtype = tf.float32), select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type)
model.summary()

# All of these variables are tensorflow tensors
# Create a boolean mask

before_loss_time = time.time()
for i in range(100):
    E, F = model(Rin2, Cha2, tf.cast(LArray2, dtype = tf.float32), select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type)
end = time.time()
print('time 22 elapsed %.4f'%(end - before_loss_time))

## Calculate the neighbor list using Numpy, which is significantly faster than using TensorFlow variables ###############################################################################
#before_loss_time = time.time()
Idx_O_A, Idx_H_A, Idx_O_R, Idx_H_R = find_and_sort_neighbors_water(pointsArray, chargesArray, L, radious_A, maxNumNeighs_O_A, maxNumNeighs_H_A, radious_R, maxNumNeighs_O_R, maxNumNeighs_H_R)
print(Idx_O_A.shape, Idx_H_A.shape, Idx_O_R.shape, Idx_H_R.shape)
#(270, 300, 16) (270, 300, 32) (270, 300, 60) (270, 300, 120)

mask_charge_O = tf.equal(chargesArray, 1)  # Create a boolean mask of shape (2, 300) for O
mask_charge_H = tf.equal(chargesArray, 2)  # Create a boolean mask of shape (2, 300) for H
for i in ["O", "H"]: 
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"Idx_{i}_{j}_{k} = []")
for m in range(pointsArray.shape[0]):  # Iterate over the batch
    for i in ["O", "H"]:
        exec(f"batch_mask_{i} = mask_charge_{i}[m]")  # Current batch mask
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"selected_{i}_{j}_{k} = tf.boolean_mask(Idx_{j}_{k}[m], batch_mask_{i})")
                exec(f"Idx_{i}_{j}_{k}.append(selected_{i}_{j}_{k})")
for i in ["O", "H"]:
    for j in ["O", "H"]:
        for k in ["A", "R"]:
            exec(f"Idx_{i}_{j}_{k} =  tf.stack(Idx_{i}_{j}_{k}, axis=0)")
            exec(f"print(Idx_{i}_{j}_{k}.shape)")


####   Test Set   ####
IdxTest_O_A, IdxTest_H_A, IdxTest_O_R, IdxTest_H_R = find_and_sort_neighbors_water(pointsTest, chargesTest, L, radious_A, maxNumNeighs_O_A, maxNumNeighs_H_A, radious_R, maxNumNeighs_O_R, maxNumNeighs_H_R)
mask_Test_O = tf.equal(chargesTest, 1)  # Create a boolean mask of shape (2, 300) for O
mask_Test_H = tf.equal(chargesTest, 2)  # Create a boolean mask of shape (2, 300) for H
for i in ["O", "H"]: 
    for j in ["O", "H"]:
        for k in ["A", "R"]:
        exec(f"IdxTest_{i}_{j}_{k} = []")
for m in range(pointsTest.shape[0]):  # Iterate over the batch
    for i in ["O", "H"]:
    exec(f"batchTest_mask_{i} = mask_Test_{i}[m]")  # Current batch mask
    for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"selectedTest_{i}_{j}_{k} = tf.boolean_mask(IdxTest_{j}_{k}[m], batchTest_mask_{i})")
                exec(f"IdxTest_{i}_{j}_{k}.append(selectedTest_{i}_{j}_{k})")
for i in ["O", "H"]:
    for j in ["O", "H"]:
        for k in ["A", "R"]:
            exec(f"IdxTest_{i}_{j}_{k} =  tf.stack(IdxTest_{i}_{j}_{k}, axis=0)")
            exec(f"print(IdxTest_{i}_{j}_{k}.shape)")

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

errorlist_energy = []
errorlist_force = []
errorlist_energy_train = []
errorlist_force_train = []
losslist = []

### optimization parameters ###
mse_loss_fn = tf.keras.losses.MeanSquaredError()
initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps = ((Nsamples//10*9)//batchSizeArray[0])*epochsPerStair,
             decay_rate = decayRate,
             staircase = True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)
loss_metric = tf.keras.metrics.Mean()

#############################################  load model  ########################################
if Test_type == "SLR2" or Load_short == 1:
  print("load Short model!")
  model.load_weights(model_load)
  model.compile()
  for layer in model.layers:
    print(f"{layer.name} is trainable: {layer.trainable}")

###################  training loop  ##################################
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")
min_test_F_err = np.Inf
min_test_E_err = np.Inf
for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = Energy_rate
  weightF = Force_rate

  x_train = (pointsArray, energyArray, forcesArray, chargesArray, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R)
  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL) # batchSizeL 2 4 8 16

  # Iterate over epochs
  for epoch in range(epochs):
  
    loss_metric.reset_state()

    # Iterate over the batches of the dataset
    for step, x_batch_train in enumerate(train_dataset):
      start_epoch_time = time.time()
      
    # x_batch_train[0] is the input, x_batch_train[1] is the energy output, and x_batch_train[2] is the force output
      loss = train_Water(model, optimizer, mse_loss_fn, x_batch_train[0], x_batch_train[3], L, select_neuron, inner_factor_A, radious_A, x_batch_train[4], x_batch_train[5], x_batch_train[6], x_batch_train[7], inner_factor_R, radious_R, x_batch_train[8], x_batch_train[9], x_batch_train[10], x_batch_train[11], Test_type, x_batch_train[1], x_batch_train[2], weightE, weightF)

      loss_metric(loss)

    pottrain, forcetrain = model(pointsArray[:10], chargesArray[:10], tf.cast(LArray2, dtype = tf.float32), select_neuron, inner_factor_A, radious_A, Idx_O_O_A[:10], Idx_H_O_A[:10], Idx_O_H_A[:10], Idx_H_H_A[:10], inner_factor_R, radious_R, Idx_O_O_R[:10], Idx_H_O_R[:10], Idx_O_H_R[:10], Idx_H_H_R[:10], Test_type) 
    mae = tf.keras.losses.MeanAbsoluteError()
    err_train = mae(forcetrain, forcesArray[:10,:Npoints,:])
    err_ener_train = mae(pottrain, energyArray[:10])

    potPred, forcePred = model(pointsTest, chargesTest, L, select_neuron,  inner_factor_A, radious_A, IdxTest_O_O_A, IdxTest_H_O_A, IdxTest_O_H_A, IdxTest_H_H_A, inner_factor_R, radious_R, IdxTest_O_O_R, IdxTest_H_O_R, IdxTest_O_H_R, IdxTest_H_H_R, Test_type)
    
    err = mae(forcePred, forcesTest)
    err_ener = mae(potPred, energyTest) 

    if err < min_test_F_err:
      min_test_F_err = err
      model.save_weights(saveFolder+time_name0+'_best_model.h5', save_format='h5')
      print("Relative Error in the trained forces is " +str(err_train.numpy()))
      print("Relative Error in the test forces is " +str(err.numpy()))

    
    # save the error
    errorlist_force.append(err.numpy())  # Append the test force error to the error list
    with open(saveFolder+time_name0+'_error_force_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force)
    errorlist_energy.append(err_ener.numpy())  # Append the test energy error to the error list
    with open(saveFolder+time_name0+'_error_energy_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy) 
    errorlist_force_train.append(err_train.numpy())
    with open(saveFolder+time_name0+'_error_force_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force_train)
    errorlist_energy_train.append(err_ener_train.numpy())  # Append the training energy error to the error list
    with open(saveFolder+time_name0+'_error_energy_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy_train) 
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # decay learning rate 
    
    #lrStr = str(optimizer._decayed_lr('float32').numpy())
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
  ### Get timestamp ###
  str_num = str(cycle)
  
  weights = model.layers[3].get_weights()  # Retrieve the weights and biases of the 4th layer
  print(weights)
  model.save_weights(saveFolder+time_name+'my_model_'+str_num+'.h5', save_format='h5')

