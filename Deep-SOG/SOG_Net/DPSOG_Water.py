import tensorflow as tf
import time
from SOG_Net.lrc_layers import SOG_3DmixedSpecies_Water
from SOG_Net.utilities import pyramidLayer, DenseLayer
from SOG_Net.Neighbor import gen_coor_3d_species_water_Force_Double_List

class DPSOG_Water(tf.keras.Model):

  """Combines the encoder and decoder into an end-to-end model for training."""
  def __init__(self,
               Npoints,
               maxNumNeighs_O_A = 14,
               maxNumNeighs_H_A = 28,
               maxNumNeighs_O_R = 60,
               maxNumNeighs_H_R = 120,
               L = 14.4088,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               NpointsFourier = 500, 
               fftChannels = 4,
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               name='dpSOG_Water',
               **kwargs):
    super(DPSOG_Water, self).__init__(name=name, **kwargs)
    self.Npoints = Npoints
    # maximum number of neighbors
    self.maxNumNeighs_O_A = maxNumNeighs_O_A
    self.maxNumNeighs_H_A = maxNumNeighs_H_A
    self.maxNumNeighs_O_R = maxNumNeighs_O_R
    self.maxNumNeighs_H_R = maxNumNeighs_H_R
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    self.fftChannels = fftChannels

    # we may need to use the tanh here
    self.layerPyramid_O_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network
    self.layerPyramid_H_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network
    self.layerPyramid_O_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network
    self.layerPyramid_H_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network

    self.layerPyramidDir_O_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network
    self.layerPyramidDir_H_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network
    self.layerPyramidDir_O_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network
    self.layerPyramidDir_H_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network

    self.layerPyramid_Q = pyramidLayer([16, 16, 16], actfn = tf.nn.tanh)
    self.lrc_layer = SOG_3DmixedSpecies_Water(fftChannels, NpointsFourier, L)

    self.fittingNetwork = pyramidLayer(fittingDim, actfn = tf.nn.tanh) # Fit network
    self.linfitNet = DenseLayer(1)  # Ener network  
  
  @tf.function
  def call(self, inputs, charge_index, L, select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type = "SR"):
    
    # we watch the inputs 
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(inputs)  
      
      # For simulation
      RI_A, Sij_O_O_A, Sij_H_O_A, Sij_O_H_A, Sij_H_H_A, Sij_O_O_R, Sij_H_O_R, Sij_O_H_R, Sij_H_H_R = gen_coor_3d_species_water_Force_Double_List(inputs, L, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R)
      
      GI_A_O_O = self.layerPyramid_O_O(Sij_O_O_A)
      GI_A_H_O = self.layerPyramid_H_O(Sij_H_O_A)
      GI_A_O_H = self.layerPyramid_O_H(Sij_O_H_A)
      GI_A_H_H = self.layerPyramid_H_H(Sij_H_H_A)
      
      # Concatenate along the second dimension (axis=1)
      GI_A_O = tf.concat([GI_A_O_O, GI_A_O_H], axis=2)
      GI_A_H = tf.concat([GI_A_H_O, GI_A_H_H], axis=2)
      GI_A = tf.concat([GI_A_O, GI_A_H], axis=1)
      
      RiaGia = tf.matmul(RI_A, GI_A, transpose_a=True)

      # Take the first select_neuron dimensions
      Da = tf.matmul(RiaGia[:,:,:,:select_neuron], RiaGia, transpose_a=True)/((self.maxNumNeighs_O_A+self.maxNumNeighs_H_A)**2)

      GI_R_O_O = self.layerPyramidDir_O_O(Sij_O_O_R)
      GI_R_H_O = self.layerPyramidDir_H_O(Sij_H_O_R)
      GI_R_O_H = self.layerPyramidDir_O_H(Sij_O_H_R)
      GI_R_H_H = self.layerPyramidDir_H_H(Sij_H_H_R)

      # Concatenate along the second dimension (axis=1)
      GI_R_O = tf.concat([GI_R_O_O, GI_R_O_H], axis=2)
      GI_R_H = tf.concat([GI_R_H_O, GI_R_H_H], axis=2)
      GI_R = tf.concat([GI_R_O, GI_R_H], axis=1)
      
      Dr = tf.reduce_mean(GI_R, axis=2)
      
      D = tf.concat([tf.reshape(Da, (Da.shape[0], Da.shape[1], Da.shape[2] * Da.shape[3])), Dr], axis=2) # (2, 25, 500)

      Fit = self.fittingNetwork(D)
      Ea = self.linfitNet(Fit)
      Ea = tf.squeeze(Ea, axis=[2])
      Ea = tf.reduce_sum(Ea, axis=[-1])
      
      #if Test_type!="SR":
      Q = self.layerPyramid_Q(D)
      
      long_range_coord, Force_long = self.lrc_layer(inputs, Q) # (2, 25, 5, 16)
      El = tf.reduce_sum(long_range_coord, axis=[-1])

      if Test_type=="SR":
        Energy = Ea
      elif Test_type=="LR":  
        Energy = El
      else:
        Energy = Ea + El
    
    if Test_type=="SR":
      Forces = -tape.gradient(Ea, inputs)
    elif Test_type=="LR":
      Forces = Force_long
    else:
      Forces = -tape.gradient(Ea, inputs) + Force_long

    return Energy, Forces 