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
    
    #self.layerPyramidLr = pyramidLayer(descripDim, actfn = tf.nn.tanh) # La network

    #self.layerPyramid_Q = pyramidLayer([2, 4, 8, 16, 32], actfn = tf.nn.tanh) # Q network

    self.layerPyramid_Q = pyramidLayer_ThreeBodyQ_DNN([96,200,200, 1], actfn = tf.nn.tanh)
    #self.layerPyramid_Q = pyramidLayer_ThreeBodyQ([3,2,2,2], actfn = tf.nn.tanh)

    #self.layerPyramid_Q = PyramidLayerWithConv([32, 32, 32, 32], actfn = tf.nn.tanh) 

    self.lrc_layer = SOG_3DmixedSpecies_dimer(fftChannels,NpointsFourier,xLims)
    # self.layer_pyramid_lr = pyramidLayer_Q(descripDim, actfn = tf.nn.tanh)
    
    self.fittingNetwork = pyramidLayer(fittingDim, actfn = tf.nn.tanh) # Fit Da network
    #self.fittingNetworkDrLr = pyramidLayer(fittingDim, actfn = tf.nn.tanh) # Fit DrLr network
    self.linfitNet = DenseLayer(1)  # Ener Da network
    #self.linfitNetdrlr = DenseLayer(1)  # Ener DrLr network  
  
  @tf.function
  def call(self, inputs, charge_index, neigh_list, radious, Test_type = "SR"):
    with tf.GradientTape() as tape:
      # we watch the inputs
      tape.watch(inputs) 
      #print("Network")   
      #print(inputs.dtype)
      #print(radious.shape)
      RIa, SRij = gen_coor_3d_species_three_body_dimer(inputs, neigh_list, self.L, 3*radious/4, radious, self.av, self.std)
      #print(RIa.shape, SRij.shape) # (2, 25, 24, 4) (2, 25, 24, 1)
      # RIa=tf.cast(RIa,tf.float32)
      Gia = self.layerPyramid(SRij) # (2, 25, 24, 100)
      #print(Gia.shape)
      #print(RIa.dtype,Gia.dtype)
      RiaGia = tf.matmul(RIa, Gia, transpose_a=True) # (2, 25, 4, 100)
      #print(RiaGia.shape)
      
      # 取前4个维度
      Da = tf.matmul(RiaGia[:,:,:,:80], RiaGia, transpose_a=True)/(self.maxNumNeighs**2) # (2, 25, Ml = 32, 32)
      #print(Da.shape, self.maxNumNeighs**2)

      Gi = self.layerPyramidDir(SRij)
      Dr = tf.reduce_mean(Gi, axis=2)#/self.maxNumNeighs # (2, 25, 32)
      #print(Dr.shape)

      # 径向+角向
      D = tf.reshape(Da, (Da.shape[0],Da.shape[1],Da.shape[2]*Da.shape[3]))
      #print(D.shape)
      Dcona = tf.concat([D, Dr], axis=-1)
      Fit = self.fittingNetwork(Dcona) # (2, 25, 64)
      Ea = self.linfitNet(Fit) # (2, 25, 1)
      Ea = tf.reduce_sum(Ea,axis=[1,2])
      
      # 生成电荷Q
      #Q = self.layerPyramid_Q(tf.reshape(RIa,(RIa.shape[0],RIa.shape[1],-1)))
      Q = self.layerPyramid_Q(tf.reshape(RIa,(RIa.shape[0],RIa.shape[1],RIa.shape[2]*RIa.shape[3])))
      #Q = self.layerPyramid_Q(tf.reshape(Dcona,(Dcona.shape[0],Dcona.shape[1],-1)))
      #Q = self.layerPyramid_Q(tf.reshape(Dcona,(Dcona.shape[0],Dcona.shape[1],-1)))


      #Sprint(Q.shape)
      #Q = tf.reshape(Q,(Dcona.shape[0],Dcona.shape[1],-1))
      #print(Q.shape)
      long_range_coord = self.lrc_layer(inputs, Q) # (2, 25, 5, 16)
      El = tf.reduce_sum(long_range_coord, axis=[-1])
      
      #Lr = tf.reshape(long_range_coord, (long_range_coord.shape[0], long_range_coord.shape[1], -1, 1)) # (2, 25, 256, 1)
      #print(La.shape)
      #Lr = self.layerPyramidLr(Lr)
      #Lr = tf.reduce_mean(Lr, axis=2) # (2, 25, 32) 
      #print(La.shape)
      
      # DrLr = tf.concat([Dr, Lr], axis = 2)
      # El = self.fittingNetworkDrLr(DrLr)
      # El = self.linfitNetdrlr(El)
      # El = tf.squeeze(El,axis=-1)
      #print(El.shape) 
      # (2, 25, 5, 16)

      if Test_type=="SR":
        Energy = Ea
      elif Test_type=="LR":  
        Energy = El
      else:
        #Mid = self.linfitNetdrlr(tf.stack([Ea, El], axis=1))
        #Energy = tf.reduce_sum(Mid, axis=1)
        Energy = Ea + El

      #Energy = tf.reduce_sum(Energy,axis=-1)

      #### 长短一起fitting ####
      #if Test_type=="SR":
      #  D = tf.concat([D, tf.zeros_like(tf.reshape(long_range_coord, (long_range_coord.shape[0],long_range_coord.shape[1],long_range_coord.shape[2]*long_range_coord.shape[3])))], axis=2) # (2, 25, 580)
      #elif Test_type=="LR":
      #  D = tf.concat([tf.zeros_like(D), tf.reshape(long_range_coord, (long_range_coord.shape[0],long_range_coord.shape[1],long_range_coord.shape[2]*long_range_coord.shape[3]))], axis=2) # (2, 25, 5, 116)
      #else:
      #  D = tf.concat([D, tf.reshape(long_range_coord, (long_range_coord.shape[0],long_range_coord.shape[1],long_range_coord.shape[2]*long_range_coord.shape[3]))], axis=2) # (2, 25, 5, 116)
      #print(D.shape)
      #Fit = self.fittingNetwork(D) # (2, 25, 64)
      #Energy = self.linfitNet(Fit) # (2, 25, 1)
      #Energy = tf.reduce_sum(Energy, axis = [1,2]) # (2,)
      #print(Fit.shape)

      #Fit = tf.reshape(Fit, (Fit.shape[0], Fit.shape[1], Fit.shape[2]*Fit.shape[3])) # (2, 25, 320)
      #print(Fit.shape)
    
      #print(Energy.shape)

      #print(Energy.shape)
      #if Test_type=="SR":
      #  Energy = D
      #elif Test_type=="LR":
      #  Energy = long_range_coord
      #else: 
      #  Energy = D + long_range_coord   
      #Energy = tf.reduce_sum(Energy, axis = 1)
    Forces = -tape.gradient(Energy, inputs)
    return Energy, Forces
