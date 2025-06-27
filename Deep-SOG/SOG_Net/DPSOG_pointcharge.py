import tensorflow as tf
import time
from SOG_Net.utilities import gen_coor_3d_Q
from SOG_Net.lrc_layers import SOG_3D_pointcharge
from SOG_Net.utilities import pyramidLayer, DenseLayer

class DPSOG_pointcharge(tf.keras.Model):

  """Combines the encoder and decoder into an end-to-end model for training."""
  def __init__(self,
               Npoints,
               L, 
               maxNumNeighs = 4,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               NpointsFourier = 500, 
               fftChannels = 4,
               xLims = [0.0, 10.0],
               name='DPSOG_pointcharge',
               **kwargs):
    super(DPSOG_pointcharge, self).__init__(name=name, **kwargs)
    self.L = L
    self.Npoints = Npoints
    # maximum number of neighbors
    self.maxNumNeighs = maxNumNeighs
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    self.fftChannels = fftChannels
    # we may need to use the tanh here
    self.layerPyramid = pyramidLayer(descripDim, actfn = tf.nn.tanh)    
    self.layerPyramidDir = pyramidLayer(descripDim, actfn = tf.nn.tanh)
    
    self.layerPyramid_Q = pyramidLayer(descripDim, actfn = tf.nn.tanh)
    self.layerPyramidDir_Q = pyramidLayer(descripDim, actfn = tf.nn.tanh)

    self.lrc_layer = SOG_3D_pointcharge(fftChannels,NpointsFourier,xLims)
    self.layer_pyramid_lr = pyramidLayer(descripDim, actfn = tf.nn.tanh)
    self.fittingNetwork = pyramidLayer(fittingDim, actfn = tf.nn.tanh)
    self.linfitNet = DenseLayer(1)    

  @tf.function
  def call(self, inputs, charge, neigh_list):

    with tf.GradientTape() as tape:
      # we watch the inputs 
      tape.watch(inputs)

      gen_coordinates = gen_coor_3d_Q(inputs,charge,  neigh_list, self.L, self.av, self.std)
      L1 = self.layerPyramid(gen_coordinates[:,:1])*gen_coordinates[:,:1]
      L2 = self.layerPyramidDir(gen_coordinates[:,1:4])*gen_coordinates[:,:1]
      L1_Q = self.layerPyramid_Q(gen_coordinates[:,4:5])*gen_coordinates[:,:1]
      L2_Q = self.layerPyramidDir_Q(gen_coordinates[:,5:])*gen_coordinates[:,:1]
      LL = tf.concat([L1, L2, L1_Q, L2_Q], axis = 1)
      Dtemp = tf.reshape(LL, (-1, self.maxNumNeighs, 4*self.descriptorDim))
      D = tf.reduce_sum(Dtemp, axis = 1)
      long_range_coord = self.lrc_layer(inputs, charge)
      long_range_coord2 = tf.reshape(long_range_coord, (-1, self.fftChannels))
      L3   = self.layer_pyramid_lr(long_range_coord2)
      DLongRange = tf.concat([D, L3], axis = 1)
      F2 = self.fittingNetwork(DLongRange)
      F = self.linfitNet(F2)
      F_reshape = tf.reshape(F, (-1, self.Npoints))
      Energy = tf.reduce_sum(F_reshape, keepdims = True, axis = 1)
      
    Forces = -tape.gradient(Energy, inputs)
    return Energy, Forces