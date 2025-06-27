import tensorflow as tf
from SOG_Net.lrc_layers import SOG_3DmixedSpecies_dimer
from SOG_Net.utilities import pyramidLayer,DenseLayer,pyramidLayer_ThreeBodyQ_DNN
from SOG_Net.Neighbor import gen_coor_3d_species_three_body_dimer

class DPSOG_dimer(tf.keras.Model):
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
               name='DPSOG_dimer',
               **kwargs):
    super(DPSOG_dimer, self).__init__(name=name, **kwargs)
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
    self.layerPyramid = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network
    self.layerPyramidDir = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network
    self.layerPyramid_Q = pyramidLayer_ThreeBodyQ_DNN([96,200,200, 1], actfn = tf.nn.tanh)
    self.lrc_layer = SOG_3DmixedSpecies_dimer(fftChannels,NpointsFourier,xLims)
    self.fittingNetwork = pyramidLayer(fittingDim, actfn = tf.nn.tanh) # Fit Da network
    self.linfitNet = DenseLayer(1)  # Ener Da network
  
  @tf.function
  def call(self, inputs, charge_index, neigh_list, radious, Test_type = "SR"):
    with tf.GradientTape() as tape:
      # we watch the inputs
      tape.watch(inputs) 
      RIa, SRij = gen_coor_3d_species_three_body_dimer(inputs, neigh_list, self.L, 3*radious/4, radious, self.av, self.std)
      Gia = self.layerPyramid(SRij)
      RiaGia = tf.matmul(RIa, Gia, transpose_a=True)

      # Take the first 4 dimensions
      Da = tf.matmul(RiaGia[:,:,:,:80], RiaGia, transpose_a=True)/(self.maxNumNeighs**2)
      Gi = self.layerPyramidDir(SRij)
      Dr = tf.reduce_mean(Gi, axis=2)

      # Radial + Angular
      D = tf.reshape(Da, (Da.shape[0],Da.shape[1],Da.shape[2]*Da.shape[3]))
      Dcona = tf.concat([D, Dr], axis=-1)
      Fit = self.fittingNetwork(Dcona) # (2, 25, 64)
      Ea = self.linfitNet(Fit) # (2, 25, 1)
      Ea = tf.reduce_sum(Ea,axis=[1,2])
      
      # Generate charges Q
      Q = self.layerPyramid_Q(tf.reshape(RIa,(RIa.shape[0],RIa.shape[1],RIa.shape[2]*RIa.shape[3])))
      long_range_coord = self.lrc_layer(inputs, Q) # (2, 25, 5, 16)
      El = tf.reduce_sum(long_range_coord, axis=[-1])

      if Test_type=="SR":
        Energy = Ea
      elif Test_type=="LR":  
        Energy = El
      else:
        Energy = Ea + El

    Forces = -tape.gradient(Energy, inputs)
    return Energy, Forces
