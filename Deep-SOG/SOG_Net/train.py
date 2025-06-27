import tensorflow as tf
import numpy as np 
import sys
from numba import jit 
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
from numba import vectorize, float32, njit

## Training functions ##
@tf.function
def train_pointcharge(model, optimizer, loss, inputs, neigh_list, charge, output_E, output_f, weight_e, weight_f):

  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE, predF = model(inputs, charge, neigh_list, training=True)

    # fidelity loss usin mse
    lossE = loss(predE, output_E)
    lossF = loss(predF, output_f)/output_f.shape[-1]

    total_loss = weight_e*lossE + weight_f*lossF

  gradients = tape.gradient(total_loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss, gradients


# @tf.function
def train_dimer(model, optimizer, loss, inputs, neigh_list, charge, radious, output_E, output_f, weight_e, weight_f,Test_type):

  with tf.GradientTape() as tape:

    # we use the model the predict the outcome
    predE, predF = model(inputs, charge, neigh_list, radious, Test_type, training=True)
    
    lossE = loss(predE, output_E)
    lossF = loss(predF, output_f)/output_f.shape[-1]
    total_loss = weight_e * lossE + weight_f * lossF
  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  '''
  for weight in model.trainable_weights:
    print(f"Weight: {weight.name}, dtype: {weight.dtype}")
  print("gradients dtype")
  for grad in gradients:
    if grad is not None:
        print(grad.dtype)
  '''
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss, gradients


@tf.function
def train_Water(model, optimizer, loss, inputs, charge, L, select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type, output_E, output_f, weight_e, weight_f):
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE, predF = model(inputs, charge, L, select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type, training=True)
    lossE = loss(predE, output_E)
    lossF = loss(predF, output_f)/output_f.shape[-1]
      
    total_loss = weight_e * lossE + weight_f * lossF

  gradients = tape.gradient(total_loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss #, gradients 