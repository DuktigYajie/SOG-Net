import tensorflow as tf
import numpy as np 
import time
import finufft
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow_nufft as tfft

@tf.function  
def gaussianPer(x, tau, L = 2*np.pi):
  x = tf.cast(x, tf.float32)
  L = tf.cast(L, tf.float32)
  tau = tf.cast(tau, tf.float32)
  return tf.exp( -tf.square(x)/(4*tau)) + \
         tf.exp( -tf.square(x-L)/(4*tau)) + \
         tf.exp( -tf.square(x+L)/(4*tau))

@tf.function 
def gaussianTot(x, tau, L = 2*np.pi):
  return tf.exp( -x/(4*tau))

@tf.function 
def gaussianDeconv2D(kx, ky, tau):
  return (np.pi/tau)*tf.exp((tf.square(kx) + tf.square(ky))*tau)

@tf.function 
def gaussianDeconv3D(kx, ky, kz, tau):
  return tf.sqrt(np.pi/tau)**3*tf.exp((tf.square(kx) + tf.square(ky) + tf.square(kz))*tau)

@tf.function 
def gaussianDeconv3DTot(ksquare, tau):
  return tf.sqrt(np.pi/tau)**3*tf.exp(ksquare*tau)

@tf.function 
def blackbox(kx, ky, kz, beta,lam,h_l):
  #return (beta)*tf.exp(-(tf.square(kx) + tf.square(ky) + tf.square(kz))/(tf.exp(-lam))**2) #考虑zero mode
  squared_sum = tf.square(kx) + tf.square(ky)+ tf.square(kz)
  condition = tf.equal(squared_sum, 0)
  result = beta * tf.exp(-squared_sum / (tf.exp(-lam))**2)
  return tf.where(condition, 0.0, result) #Do not consider zero frequency mode

class SOG_3D_pointcharge(tf.keras.layers.Layer):
  def __init__(self, nChannels, NpointsMesh, xLims):
    super(SOG_3D_pointcharge, self).__init__()
    self.nChannels = nChannels
    self.bandwidth_num = 12
    self.NpointsMesh = NpointsMesh 
    
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    self.xLims = xLims

    print(xLims)

    self.L = np.abs(self.xLims[1] - self.xLims[0])

    self.tau = tf.constant(16*(self.L/(2*np.pi*NpointsMesh))**2, 
                           dtype = tf.float32)# the size of the mollifications

    self.kGrid = tf.constant((2*np.pi/self.L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    self.kx_grid, self.ky_grid, self.kz_grid = tf.meshgrid(self.kGrid, self.kGrid, self.kGrid) 

    # we need to define a mesh betwen xLims[0] and xLims[1]
    self.xGrid =  tf.constant(np.linspace(xLims[0], xLims[1], NpointsMesh+1)[:-1], dtype = tf.float32)

    self.x_grid, self.y_grid, self.z_grid = tf.meshgrid(self.xGrid, self.xGrid, self.xGrid) 

    self.bandwidth = tf.linspace(-1.5, 2.3, self.bandwidth_num) #exponential #

  def build(self, input_shape):

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here
    
    self.shift = []

    a = self.bandwidth.numpy()
    for ii in range(self.bandwidth_num):
      init = tf.constant_initializer(a[ii])
      self.shift.append(self.add_weight(name="std_"+str(ii),
                       initializer=init,
                       shape=[1,]))
                       
    self.amplitud = []
    
    for ii in range(self.bandwidth_num):
      self.amplitud.append(self.add_weight(name="bias_"+str(ii),
                       initializer=tf.initializers.ones(), shape=[1,]))#initializer=init, shape=[1,])) #initializer=init, shape=[1,])) #

  @tf.function
  def call(self, input, charge):
      start_time = time.time() # Start timing

      energies = []

      multiplier = 0 * self.kx_grid
      squared_sum = tf.square(self.kx_grid) + tf.square(self.ky_grid) + tf.square(self.kz_grid)
      condition = tf.equal(squared_sum, 0)
      self_shift = tf.convert_to_tensor(self.shift, dtype=tf.float32)
      self_amplitud = tf.convert_to_tensor(self.amplitud, dtype=tf.float32)

      min_term = - 1 / tf.square( tf.exp(-self_shift[:,0])) #exponential

      min_term = tf.expand_dims(tf.expand_dims(tf.expand_dims(min_term,0),0),0)
      a = tf.expand_dims(squared_sum,-1)
      multiplier = tf.expand_dims(tf.expand_dims(tf.expand_dims(self_amplitud[:,0],0),0),0) * tf.exp(a*min_term)
      multiplier = tf.reduce_sum(multiplier, axis=-1)      
      multiplier= tf.where(condition, 0.0, multiplier)
      multiplier = tf.expand_dims(multiplier, 0)
      multiplierRe = tf.math.real(multiplier)
      V = tf.cast(self.L, tf.float32) * tf.cast(self.L, tf.float32) * tf.cast(self.L, tf.float32)
      diag_sum = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(multiplier,-1),-1),-1)/(V*2)
      charge = tf.cast(charge, dtype=tf.float32) 
      diag_sum = charge * diag_sum  
      charge_complex=tf.complex(charge, 0.00)
      input_te=tf.constant(2*np.pi,dtype = tf.float32)/tf.cast(self.L, tf.float32)*(input-self.L/2)
      recon = tfft.nufft(charge_complex, input_te, grid_shape=self.x_grid.shape,transform_type='type_1', fft_direction='forward', tol=1e-05)     
      Rerecon=tf.math.real(recon)
      Imrecon=tf.math.imag(recon)
      multRe = tf.multiply(multiplierRe, Rerecon)
      multIm = tf.multiply(multiplierRe, Imrecon)
      mult_fft = tf.complex(multRe, multIm)
      Ifftcon = tf.math.real(tfft.nufft(mult_fft, input_te, [], transform_type='type_2', fft_direction='backward', tol=1e-05)/tf.complex(2*V,0.0))
      energy = (Ifftcon - diag_sum) * charge
      energies=[]
      energies.append(tf.expand_dims(energy, axis=-1))
      energy = tf.concat(energies, axis=-1)

###

      return energy

# No self-energy term included, including the zeroth-order term
class SOG_3DmixedSpecies_dimer(tf.keras.layers.Layer):
  def __init__(self, nChannels, NpointsMesh, xLims):
    super(SOG_3DmixedSpecies_dimer, self).__init__()
    self.nChannels = nChannels
    self.bandwidth_num = 12
    self.NpointsMesh = NpointsMesh 
    
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1
    self.xLims = xLims
    print(xLims)
    self.L = np.abs(self.xLims[1] - self.xLims[0])
    self.tau = tf.constant(16*(self.L/(2*np.pi*NpointsMesh))**2, 
                           dtype = tf.float32)
    self.kGrid = tf.constant((2*np.pi/self.L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    self.kx_grid, self.ky_grid, self.kz_grid = tf.meshgrid(self.kGrid, self.kGrid, self.kGrid) 

    # we need to define a mesh betwen xLims[0] and xLims[1]
    self.xGrid =  tf.constant(np.linspace(xLims[0], xLims[1], NpointsMesh+1)[:-1], dtype = tf.float32)
    self.x_grid, self.y_grid, self.z_grid = tf.meshgrid(self.xGrid, self.xGrid, self.xGrid) 

    self.bandwidth = tf.linspace(0.0, 1.8, self.bandwidth_num) #exponential #


  def build(self, Q):

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here
    
    self.shift_1 = []
    a = self.bandwidth.numpy()
    init = tf.constant_initializer(a)
    self.shift_1 = self.add_weight(name="std_1", dtype=tf.float32, initializer=init, shape=(self.bandwidth_num), trainable=True)
    self.amplitud_1 = [] 
    self.amplitud_1 = self.add_weight(name="bias_1", dtype=tf.float32, initializer=tf.initializers.ones(), shape=(self.bandwidth_num), trainable=True)
    tf.print(self.shift_1)
    tf.print(self.amplitud_1)

  @tf.function
  def call(self, input, charge):
      
      V = tf.cast(self.L, tf.float32) * tf.cast(self.L, tf.float32) * tf.cast(self.L, tf.float32)
      squared_sum = tf.square(self.kx_grid) + tf.square(self.ky_grid) + tf.square(self.kz_grid)
      condition = tf.equal(squared_sum, 0)
      a = tf.expand_dims(squared_sum,-1)
      min_term = - 1 / tf.exp(-2*self.shift_1[:]) #exponential
      min_term = tf.expand_dims(tf.expand_dims(tf.expand_dims(min_term, 0), 0),0)
      multiplier = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.amplitud_1[:],0),0),0) * tf.exp(a*min_term)
      multiplier = tf.reduce_sum(multiplier, axis=-1)      
      multiplier = tf.where(condition, 0.0, multiplier) # 不管zero mode的这部分 也加上去
      print("with zero")
      multiplier = tf.expand_dims(multiplier, 0)
      diag_sum = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(multiplier,-1),-1),-1)/(V*2)
      multiplierRe = tf.math.real(multiplier)

      charge_complex = tf.reshape(charge,(charge.shape[0], charge.shape[1], -1))
      charge_complex = tf.complex(charge_complex, 0.0)
      print(charge_complex.shape)
      
      input_te = tf.constant(2*np.pi,dtype = tf.float32)/tf.cast(self.L, tf.float32)*(input-self.L/2)
      
      energy = tf.zeros((input.shape[0], input.shape[1]), dtype=tf.float32)

      for i in range(charge_complex.shape[-1]):
         charge_select = charge_complex[:,:,i]
         grid_shape = tf.convert_to_tensor(self.x_grid.shape, dtype=tf.int32)
         recon = tfft.nufft(charge_select, input_te, grid_shape=grid_shape, transform_type='type_1', fft_direction='forward', tol=1e-05)  
         Rerecon = tf.math.real(recon)
         Imrecon = tf.math.imag(recon)
         multRe = tf.multiply(multiplierRe, Rerecon)
         multIm = tf.multiply(multiplierRe, Imrecon)
         mult_fft = tf.complex(multRe, multIm)
         Ifftcon = tf.math.real(tfft.nufft(mult_fft, input_te, [], transform_type='type_2', fft_direction='backward', tol=1e-05))/(2*V) # 计算到全部粒子位置再乘系数相加
         energy = energy + tf.math.real(charge_select) * (Ifftcon - diag_sum)
      
      return energy
      
class SOG_3DmixedSpecies_Water(tf.keras.layers.Layer):
  def __init__(self, nChannels, NpointsMesh, L):
    super(SOG_3DmixedSpecies_Water, self).__init__()
    self.nChannels = nChannels
    self.bandwidth_num = 6
    self.NpointsMesh = NpointsMesh 
    
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1 
 
    self.kGrid = tf.constant((2*np.pi/L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    self.kx_grid, self.ky_grid, self.kz_grid = tf.meshgrid(self.kGrid, self.kGrid, self.kGrid, indexing='ij')
    
    self.V = L * L * L
    self.L = L

    squared_sum = tf.square(self.kx_grid) + tf.square(self.ky_grid) + tf.square(self.kz_grid)
    
    self.condition = tf.equal(squared_sum, 0)
    self.squ_exp = tf.cast(tf.expand_dims(squared_sum, -1), dtype=tf.float32)
    self.grid_shape = tf.convert_to_tensor(self.kx_grid.shape, dtype=tf.int32)

    self.bandwidth = tf.linspace(-0.5, 1.8, self.bandwidth_num) #exponential #

   # Called the first time the layer is invoked

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here
    
    a = self.bandwidth.numpy()
    init = tf.constant_initializer(a)
    self.shift_O_O = []
    self.shift_O_O = self.add_weight(name="std_O_O", dtype=tf.float32, initializer=init, shape=(self.bandwidth_num), trainable=True)
    self.amplitud_O_O = [] 
    self.amplitud_O_O = self.add_weight(name="bias_O_O", dtype=tf.float32, initializer=tf.initializers.ones(), shape=(self.bandwidth_num), trainable=True)

    self.shift_O_H = []
    self.shift_O_H = self.add_weight(name="std_O_H", dtype=tf.float32, initializer=init, shape=(self.bandwidth_num), trainable=True)
    self.amplitud_O_H = [] 
    self.amplitud_O_H = self.add_weight(name="bias_O_H", dtype=tf.float32, initializer=tf.initializers.ones(), shape=(self.bandwidth_num), trainable=True)

    self.shift_H_O = []
    self.shift_H_O = self.add_weight(name="std_H_O", dtype=tf.float32, initializer=init, shape=(self.bandwidth_num), trainable=True)
    self.amplitud_H_O = [] 
    self.amplitud_H_O = self.add_weight(name="bias_H_O", dtype=tf.float32, initializer=tf.initializers.ones(), shape=(self.bandwidth_num), trainable=True)

    self.shift_H_H = []
    self.shift_H_H = self.add_weight(name="std_H_H", dtype=tf.float32, initializer=init, shape=(self.bandwidth_num), trainable=True)
    self.amplitud_H_H = [] 
    self.amplitud_H_H = self.add_weight(name="bias_H_H", dtype=tf.float32, initializer=tf.initializers.ones(), shape=(self.bandwidth_num), trainable=True)

  @tf.function #(jit_compile=True)
  def call(self, input, charge):
      start_time = time.time()
      
      Nsamples, Npoints, Dimenions = input.shape
      Npoints_Divide_3 = tf.cast(Npoints/3, tf.int32)

      energy = tf.zeros((input.shape[0], input.shape[1]), dtype=tf.float32)
      charge_complex = tf.reshape(charge, (charge.shape[0], charge.shape[1], -1))
      charge_complex = tf.complex(charge_complex, 0.0) # (4, 300, 16)
      
      ####  Fourier Multiplier 1 ####
      min_term_O_O = - 1 / tf.exp(-2*self.shift_O_O[:]) #exponential
      min_term_O_O = tf.expand_dims(tf.expand_dims(tf.expand_dims(min_term_O_O, 0), 0), 0)
      multiplier_O_O = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.amplitud_O_O[:],0),0),0) * tf.exp(self.squ_exp * min_term_O_O)
      multiplier_O_O = tf.reduce_sum(multiplier_O_O, axis=-1)
      multiplier_O_O = tf.expand_dims(multiplier_O_O, 0)
      multiplierRe_O_O = tf.math.real(multiplier_O_O) # (1, 21, 21, 21)
      
      ####  Fourier Multiplier 2 ####
      min_term_O_H = - 1 / tf.exp(-2*self.shift_O_H[:]) #exponential
      min_term_O_H = tf.expand_dims(tf.expand_dims(tf.expand_dims(min_term_O_H, 0), 0), 0)
      multiplier_O_H = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.amplitud_O_H[:],0),0),0) * tf.exp(self.squ_exp * min_term_O_H)
      multiplier_O_H = tf.reduce_sum(multiplier_O_H, axis=-1)
      multiplier_O_H = tf.where(self.condition, 0.0, multiplier_O_H)
      multiplier_O_H = tf.expand_dims(multiplier_O_H, 0)
      multiplierRe_O_H = tf.math.real(multiplier_O_H)
      
      ####  Fourier Multiplier 3 ####
      min_term_H_O = - 1 / tf.exp(-2*self.shift_H_O[:]) #exponential
      min_term_H_O = tf.expand_dims(tf.expand_dims(tf.expand_dims(min_term_H_O, 0), 0), 0)
      multiplier_H_O = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.amplitud_H_O[:],0),0),0) * tf.exp(self.squ_exp * min_term_H_O)
      multiplier_H_O = tf.reduce_sum(multiplier_H_O, axis=-1)
      multiplier_H_O = tf.where(self.condition, 0.0, multiplier_H_O)
      multiplier_H_O = tf.expand_dims(multiplier_H_O, 0)
      multiplierRe_H_O = tf.math.real(multiplier_H_O)
      
      ####  Fourier Multiplier 4 ####
      min_term_H_H = - 1 / tf.exp(-2*self.shift_H_H[:]) #exponential
      min_term_H_H = tf.expand_dims(tf.expand_dims(tf.expand_dims(min_term_H_H, 0), 0), 0)
      multiplier_H_H = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.amplitud_H_H[:], 0), 0), 0) * tf.exp(self.squ_exp * min_term_H_H)
      multiplier_H_H = tf.reduce_sum(multiplier_H_H, axis=-1)
      multiplier_H_H = tf.where(self.condition, 0.0, multiplier_H_H)
      multiplier_H_H = tf.expand_dims(multiplier_H_H, 0)
      multiplierRe_H_H = tf.math.real(multiplier_H_H)

      ####  Calculation  ####
      input_te = tf.constant(2*np.pi,dtype = tf.float32)/self.L * (input - self.L/2)
      input_te = tf.expand_dims(input_te, axis=1) # (4, 1, 300, 3)
      transposed_charge = tf.transpose(charge_complex, perm=[0, 2, 1]) # (4, 16, 300)
      
      recon_O = tfft.nufft(transposed_charge[:,:,:Npoints_Divide_3], input_te[:,:,:Npoints_Divide_3,:], grid_shape=self.grid_shape, transform_type='type_1', fft_direction='forward', tol=1e-03)
      recon_H = tfft.nufft(transposed_charge[:,:,Npoints_Divide_3:], input_te[:,:,Npoints_Divide_3:,:], grid_shape=self.grid_shape, transform_type='type_1', fft_direction='forward', tol=1e-03)

      Rerecon_O = tf.math.real(recon_O)
      Imrecon_O = tf.math.imag(recon_O)
      Rerecon_H = tf.math.real(recon_H)
      Imrecon_H = tf.math.imag(recon_H)

      multRe_O_O = tf.multiply(multiplierRe_O_O, Rerecon_O)
      multIm_O_O = tf.multiply(multiplierRe_O_O, Imrecon_O)
      mult_fft_O_O = tf.complex(multRe_O_O, multIm_O_O)
      
      multRe_H_O = tf.multiply(multiplierRe_H_O, Rerecon_O)
      multIm_H_O = tf.multiply(multiplierRe_H_O, Imrecon_O)
      mult_fft_H_O = tf.complex(multRe_H_O, multIm_H_O)

      multRe_O_H = tf.multiply(multiplierRe_O_H, Rerecon_H)
      multIm_O_H = tf.multiply(multiplierRe_O_H, Imrecon_H)
      mult_fft_O_H = tf.complex(multRe_O_H, multIm_O_H)

      multRe_H_H = tf.multiply(multiplierRe_H_H, Rerecon_H)
      multIm_H_H = tf.multiply(multiplierRe_H_H, Imrecon_H)
      mult_fft_H_H = tf.complex(multRe_H_H, multIm_H_H)
      
      mult_fft_O = mult_fft_O_O + mult_fft_O_H
      mult_fft_H = mult_fft_H_O + mult_fft_H_H 

      Ifftcon_O = tf.math.real(tfft.nufft(mult_fft_O, input_te[:,:,:Npoints_Divide_3,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(2*self.V) # 计算到全部粒子位置再乘系数相加 This 1/2 is due to the remove of double count of pairs
      Ifftcon_H = tf.math.real(tfft.nufft(mult_fft_H, input_te[:,:,Npoints_Divide_3:,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(2*self.V) # 计算到全部粒子位置再乘系数相加 This 1/2 is due to the remove of double count of pairs
      
      Ifftcon = tf.concat([Ifftcon_O,Ifftcon_H], axis=2)
      new_value = tf.multiply(tf.math.real(transposed_charge), Ifftcon)
      energy = tf.reduce_sum(new_value, axis = 1)
      
      kx_mult_fft_O = tf.multiply(mult_fft_O, tf.complex(self.kx_grid, 0.00))
      ky_mult_fft_O = tf.multiply(mult_fft_O, tf.complex(self.ky_grid, 0.00))
      kz_mult_fft_O = tf.multiply(mult_fft_O, tf.complex(self.kz_grid, 0.00))

      kx_mult_fft_H = tf.multiply(mult_fft_H, tf.complex(self.kx_grid, 0.00))
      ky_mult_fft_H = tf.multiply(mult_fft_H, tf.complex(self.ky_grid, 0.00))
      kz_mult_fft_H = tf.multiply(mult_fft_H, tf.complex(self.kz_grid, 0.00))

      Ifftcon_x_O = tf.math.imag(tfft.nufft(kx_mult_fft_O, input_te[:,:,:Npoints_Divide_3,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(self.V) # 计算到全部粒子位置再乘系数相加
      Ifftcon_y_O = tf.math.imag(tfft.nufft(ky_mult_fft_O, input_te[:,:,:Npoints_Divide_3,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(self.V) # 计算到全部粒子位置再乘系数相加
      Ifftcon_z_O = tf.math.imag(tfft.nufft(kz_mult_fft_O, input_te[:,:,:Npoints_Divide_3,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(self.V) # 计算到全部粒子位置再乘系数相加
      
      Ifftcon_x_H = tf.math.imag(tfft.nufft(kx_mult_fft_H, input_te[:,:,Npoints_Divide_3:,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(self.V) # 计算到全部粒子位置再乘系数相加
      Ifftcon_y_H = tf.math.imag(tfft.nufft(ky_mult_fft_H, input_te[:,:,Npoints_Divide_3:,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(self.V) # 计算到全部粒子位置再乘系数相加
      Ifftcon_z_H = tf.math.imag(tfft.nufft(kz_mult_fft_H, input_te[:,:,Npoints_Divide_3:,:], [], transform_type='type_2', fft_direction='backward', tol=1e-03))/(self.V) # 计算到全部粒子位置再乘系数相加
      
      new_value_x_O = tf.multiply(tf.math.real(transposed_charge[:,:,:Npoints_Divide_3]), Ifftcon_x_O)
      new_value_y_O = tf.multiply(tf.math.real(transposed_charge[:,:,:Npoints_Divide_3]), Ifftcon_y_O)
      new_value_z_O = tf.multiply(tf.math.real(transposed_charge[:,:,:Npoints_Divide_3]), Ifftcon_z_O)
       
      new_value_x_H = tf.multiply(tf.math.real(transposed_charge[:,:,Npoints_Divide_3:]), Ifftcon_x_H)
      new_value_y_H = tf.multiply(tf.math.real(transposed_charge[:,:,Npoints_Divide_3:]), Ifftcon_y_H)
      new_value_z_H = tf.multiply(tf.math.real(transposed_charge[:,:,Npoints_Divide_3:]), Ifftcon_z_H)
 
      Force_O = tf.stack([new_value_x_O, new_value_y_O, new_value_z_O], axis=-1)
      Force_H = tf.stack([new_value_x_H, new_value_y_H, new_value_z_H], axis=-1)
      Force = tf.concat([Force_O, Force_H], axis=2)
        
      Force = tf.reduce_sum(Force, axis=1)

      return energy, Force

