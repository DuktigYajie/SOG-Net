import tensorflow as tf
import numpy as np 
import sys
from numba import jit 
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
from numba import vectorize, float32, njit

@tf.function
def gaussianSwitch(r, rs, rc):
    #tf.print(r[0,0,:], summarize=-1)
    #print(r.shape, rs, rc)
    x = (r - rs) / (rc - rs)
    
    #tf.print(x.shape,x[0,0,:])
    # 条件 1: r < rs
    cond1 = tf.less(r, rs)
    
    # 条件 2: r < rc and r >= rs
    cond2 = tf.logical_and(tf.less(r, rc), tf.greater_equal(r, rs))
    
    # 条件下的返回值
    result = tf.where(cond1, 
                      1 / r, 
                      tf.where(cond2, 
                               1 / r * (x**3 * (-6 * x**2 + 15 * x - 10) + 1), 
                               tf.zeros_like(r)))
    #tf.print(result[0,0,:], summarize=-1)
    return result

@tf.function
def gen_coor_3d_species_three_body_dimer(r_in, neigh_list, L, rs, rc, av = tf.constant([0.0, 0.0], dtype = tf.float32), std =  tf.constant([1.0, 1.0], dtype = tf.float32)):

    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # neigh_list is a (Nsample, Npoints, maxNeigh)
    
    Nsamples = r_in.shape[0]
    Npoints = r_in.shape[1]
    max_num_neighs = neigh_list.shape[-1]
    
    #print("Here")

    # define an indicator
    mask = neigh_list > -1
    #tf.print("r_in:",r_in.shape)
    # extract per_dimension the repeated and gathered entries
    r_in_rep_X  = tf.tile(tf.expand_dims(r_in[:,:,0], -1), [1 ,1, max_num_neighs])
    r_in_gath_X = tf.gather(r_in[:,:,0], neigh_list, batch_dims = 1, axis = 1)
    r_in_rep_Y  = tf.tile(tf.expand_dims(r_in[:,:,1], -1), [1 ,1, max_num_neighs] )
    r_in_gath_Y = tf.gather(r_in[:,:,1], neigh_list, batch_dims = 1, axis = 1)
    r_in_rep_Z  = tf.tile(tf.expand_dims(r_in[:,:,2], -1), [1 ,1, max_num_neighs] )
    r_in_gath_Z = tf.gather(r_in[:,:,2], neigh_list, batch_dims = 1, axis = 1)
    
    #print(r_in_gath_X.shape, r_in_rep_X.shape, mask.shape, r_in[:,:,0].shape) # (13, 25, 24) (13, 25, 24) (13, 25, 24) (13, 25)

    # compute the periodic dimension wise distance
    r_diff_X = r_in_gath_X - r_in_rep_X
    r_diff_X = r_diff_X - L *tf.round(r_diff_X/L)
    r_diff_Y = r_in_gath_Y - r_in_rep_Y
    r_diff_Y = r_diff_Y - L*tf.round(r_diff_Y/L)
    r_diff_Z = r_in_gath_Z - r_in_rep_Z
    r_diff_Z = r_diff_Z - L*tf.round(r_diff_Z/L)
    norm = tf.sqrt(tf.square(r_diff_X) + tf.square(r_diff_Y) + tf.square(r_diff_Z))
    binv = tf.math.reciprocal(norm)
    wRij = gaussianSwitch(norm, rs, rc) # 未改变顺序之前 
    bx = tf.math.multiply(r_diff_X, binv)
    by = tf.math.multiply(r_diff_Y, binv)
    bz = tf.math.multiply(r_diff_Z, binv)
    #tf.print("Neigh:",norm.shape,bx.shape,by.shape,bz.shape)
    #tf.print(norm,rs,rc)
    #tf.print(wRij.shape)
    #tf.print(norm[1,5,:],wRij[1,5,:],summarize=-1)
    #tf.print("wRij shape:",wRij.shape,wRij)
    # 把不要的近邻部分设置为0 并且对binv_safe从大到小排序
    zeroDummy = tf.zeros_like(norm)
    binv_safe = tf.where(mask, binv, zeroDummy)
    wRij_safe = tf.where(mask, wRij, zeroDummy)
    bx_safe = tf.where(mask, bx, zeroDummy)
    by_safe = tf.where(mask, by, zeroDummy)
    bz_safe = tf.where(mask, bz, zeroDummy)
    
    # 从大到小排序
    #sorted_indices = tf.argsort(binv_safe, axis=2, direction='DESCENDING')
    #binv_safe = tf.gather(binv_safe, sorted_indices, batch_dims=2, axis=2)
    #wRij_safe = tf.gather(wRij_safe, sorted_indices, batch_dims=2, axis=2)
    #bx_safe = tf.gather(bx_safe, sorted_indices, batch_dims=2, axis=2)
    #by_safe = tf.gather(by_safe, sorted_indices, batch_dims=2, axis=2)
    #bz_safe = tf.gather(bz_safe, sorted_indices, batch_dims=2, axis=2)
    
    #tf.print(binv_safe[1,4,:],summarize = -1)
    #SRIJ = wRij_safe * binv_safe
    
    #print(SRIJ.shape, bx_safe.shape) # (13, 25, 24) (13, 25, 24) 
    #RIa= tf.stack([SRIJ, SRIJ * bx_safe, SRIJ * by_safe, SRIJ * bz_safe],axis=-1)
    RIa= tf.stack([wRij_safe, wRij_safe * bx_safe, wRij_safe * by_safe, wRij_safe * bz_safe],axis=-1)
    #print(RIa.shape) # (13, 25, 24, 4)
    #tf.print("RIa shape:",RIa.shape,RIa)
    #print(av, std)
    #binv_safe = tf.where(mask, (binv-av[0])/std[0], zeroDummy)
    #bx_safe = tf.where(mask, (bx-av[1])/std[1], zeroDummy)
    #by_safe = tf.where(mask, (by-av[2])/std[2], zeroDummy)
    #bz_safe = tf.where(mask, (bz-av[3])/std[3], zeroDummy)
    #print("Srij remove")
    # 将三个张量沿着新轴拼接成一个形状为 100x100x100x3 的张量
    #stacked_tensor = tf.stack([binv_safe, bx_safe, by_safe, bz_safe], axis=-1)
    
    #concat_tensor = tf.concat([binv_safe, bx_safe, by_safe, bz_safe], axis = 2)
    #concat_tensor = tf.concat([binv, bx, by, bz], axis = 2)

    #r_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
    #                      tf.reshape(bx_safe,   (-1,1)), 
    #                      tf.reshape(by_safe,   (-1,1)),
    #                      tf.reshape(bz_safe,   (-1,1)), ], axis = 1)
    #r_total = tf.concat([tf.reshape(binv, (-1,1)), 
    #                      tf.reshape(bx, (-1,1)), 
    #                      tf.reshape(by, (-1,1)),
    #                      tf.reshape(bz, (-1,1)), ], axis = 1)
    #tf.print(r_total)
    #tf.print("shape ", binv_safe.shape, bx_safe.shape)
    return RIa, tf.expand_dims(wRij_safe,-1) # tf.expand_dims(sRij,-1)#r_total, concat_tensor #r_total

@tf.function
def gen_coor_3d_species_water_Force_Double_List(inputs, L, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R):
    
    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # neigh_list is a (Nsample, Npoints, maxNeigh)

    Nsamples, Npoints, dimensions = inputs.shape
    #tf.print("inputs.shape ", inputs.shape)
    
    Npoints_Divide_3 = tf.cast(Npoints/3, tf.int32)
    #tf.print("Npoints_Divide_3 = ", Npoints_Divide_3)

    max_num_neighs_O_A = Idx_O_O_A.shape[-1]
    max_num_neighs_H_A = Idx_O_H_A.shape[-1]
    max_num_neighs_O_R = Idx_O_O_R.shape[-1]
    max_num_neighs_H_R = Idx_O_H_R.shape[-1]

    # print(max_num_neighs_O_A, max_num_neighs_H_A, max_num_neighs_O_R, max_num_neighs_H_R)
    # 25 50 80 160

    mask_O_O_A = Idx_O_O_A > -1
    mask_H_O_A = Idx_H_O_A > -1
    mask_O_H_A = Idx_O_H_A > -1
    mask_H_H_A = Idx_H_H_A > -1
    mask_O_O_R = Idx_O_O_R > -1
    mask_H_O_R = Idx_H_O_R > -1
    mask_O_H_R = Idx_O_H_R > -1
    mask_H_H_R = Idx_H_H_R > -1

    ##########################################################################
    # extract per_dimension the repeated and gathered entries
    # we should get an index of O ##############################################################
    r_in_rep_X_O_O_A  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 0], -1), [1 ,1, max_num_neighs_O_A])

    #Idx_O_O_A = tf.cast(Idx_O_O_A, tf.int32)
    #Idx_H_O_A = tf.cast(Idx_O_O_A, tf.int32)
    #Idx_H_H_A = tf.cast(Idx_O_O_A, tf.int32)
    #Idx_O_H_A = tf.cast(Idx_O_O_A, tf.int32)

    r_in_gath_X_O_O_A = tf.gather(inputs[:, :(Npoints_Divide_3), 0], Idx_O_O_A, batch_dims = 1, axis = 1)
    r_in_rep_Y_O_O_A  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 1], -1), [1 ,1, max_num_neighs_O_A] )
    r_in_gath_Y_O_O_A = tf.gather(inputs[:, :(Npoints_Divide_3), 1], Idx_O_O_A, batch_dims = 1, axis = 1)
    r_in_rep_Z_O_O_A  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 2], -1), [1 ,1, max_num_neighs_O_A] )
    r_in_gath_Z_O_O_A = tf.gather(inputs[:, :(Npoints_Divide_3), 2], Idx_O_O_A, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_O_O_A = r_in_gath_X_O_O_A - r_in_rep_X_O_O_A
    r_diff_X_O_O_A = r_diff_X_O_O_A - L * tf.round(r_diff_X_O_O_A/L)
    r_diff_Y_O_O_A = r_in_gath_Y_O_O_A - r_in_rep_Y_O_O_A
    r_diff_Y_O_O_A = r_diff_Y_O_O_A - L * tf.round(r_diff_Y_O_O_A/L)
    r_diff_Z_O_O_A = r_in_gath_Z_O_O_A - r_in_rep_Z_O_O_A
    r_diff_Z_O_O_A = r_diff_Z_O_O_A - L * tf.round(r_diff_Z_O_O_A/L)
    norm_O_O_A = tf.sqrt(tf.square(r_diff_X_O_O_A) + tf.square(r_diff_Y_O_O_A) + tf.square(r_diff_Z_O_O_A))
    binv_O_O_A = tf.math.reciprocal(norm_O_O_A)
    sRij_O_O_A = gaussianSwitch(norm_O_O_A, inner_factor_A * radious_A, radious_A) # 未改变顺序之前 
    bx_O_O_A = tf.math.multiply(r_diff_X_O_O_A, binv_O_O_A)
    by_O_O_A = tf.math.multiply(r_diff_Y_O_O_A, binv_O_O_A)
    bz_O_O_A = tf.math.multiply(r_diff_Z_O_O_A, binv_O_O_A)    
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_O_O_A)
    sRij_O_O_A = tf.where(mask_O_O_A, sRij_O_O_A, zeroDummy)
    bx_O_O_A = tf.where(mask_O_O_A, bx_O_O_A, zeroDummy)
    by_O_O_A = tf.where(mask_O_O_A, by_O_O_A, zeroDummy)
    bz_O_O_A = tf.where(mask_O_O_A, bz_O_O_A, zeroDummy)
    RI_O_O_A = tf.stack([sRij_O_O_A, sRij_O_O_A * bx_O_O_A, sRij_O_O_A * by_O_O_A, sRij_O_O_A * bz_O_O_A], axis = -1)
    
    # print(RI_O_O_A.shape, sRij_O_O_A.shape)
    # (10, 100, 25, 4) (10, 100, 25)

    r_in_rep_X_H_O_A  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 0], -1), [1 ,1, max_num_neighs_O_A])
    r_in_gath_X_H_O_A = tf.gather(inputs[:,:,0], Idx_H_O_A, batch_dims = 1, axis = 1)
    r_in_rep_Y_H_O_A  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 1], -1), [1 ,1, max_num_neighs_O_A] )
    r_in_gath_Y_H_O_A = tf.gather(inputs[:,:,1], Idx_H_O_A, batch_dims = 1, axis = 1)
    r_in_rep_Z_H_O_A  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 2], -1), [1 ,1, max_num_neighs_O_A] )
    r_in_gath_Z_H_O_A = tf.gather(inputs[:,:,2], Idx_H_O_A, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_H_O_A = r_in_gath_X_H_O_A - r_in_rep_X_H_O_A
    r_diff_X_H_O_A = r_diff_X_H_O_A - L * tf.round(r_diff_X_H_O_A/L)
    r_diff_Y_H_O_A = r_in_gath_Y_H_O_A - r_in_rep_Y_H_O_A
    r_diff_Y_H_O_A = r_diff_Y_H_O_A - L * tf.round(r_diff_Y_H_O_A/L)
    r_diff_Z_H_O_A = r_in_gath_Z_H_O_A - r_in_rep_Z_H_O_A
    r_diff_Z_H_O_A = r_diff_Z_H_O_A - L * tf.round(r_diff_Z_H_O_A/L)
    norm_H_O_A = tf.sqrt(tf.square(r_diff_X_H_O_A) + tf.square(r_diff_Y_H_O_A) + tf.square(r_diff_Z_H_O_A))
    binv_H_O_A = tf.math.reciprocal(norm_H_O_A)
    sRij_H_O_A = gaussianSwitch(norm_H_O_A, inner_factor_A * radious_A, radious_A) # 未改变顺序之前 
    bx_H_O_A = tf.math.multiply(r_diff_X_H_O_A, binv_H_O_A)
    by_H_O_A = tf.math.multiply(r_diff_Y_H_O_A, binv_H_O_A)
    bz_H_O_A = tf.math.multiply(r_diff_Z_H_O_A, binv_H_O_A)    
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_H_O_A)
    sRij_H_O_A = tf.where(mask_H_O_A, sRij_H_O_A, zeroDummy)
    bx_H_O_A = tf.where(mask_H_O_A, bx_H_O_A, zeroDummy)
    by_H_O_A = tf.where(mask_H_O_A, by_H_O_A, zeroDummy)
    bz_H_O_A = tf.where(mask_H_O_A, bz_H_O_A, zeroDummy)
    RI_H_O_A = tf.stack([sRij_H_O_A, sRij_H_O_A * bx_H_O_A, sRij_H_O_A * by_H_O_A, sRij_H_O_A * bz_H_O_A], axis = -1)

    # print(RI_H_O_A.shape, sRij_H_O_A.shape)
    # (10, 200, 25, 4) (10, 200, 25)
    #tf.print(Idx_H_O_A.shape,Idx_H_O_A)
    
    r_in_rep_X_O_H_A  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 0], -1), [1 ,1, max_num_neighs_H_A])
    r_in_gath_X_O_H_A = tf.gather(inputs[:,:,0], Idx_O_H_A, batch_dims = 1, axis = 1)
    r_in_rep_Y_O_H_A  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 1], -1), [1 ,1, max_num_neighs_H_A] )
    r_in_gath_Y_O_H_A = tf.gather(inputs[:,:,1], Idx_O_H_A, batch_dims = 1, axis = 1)
    r_in_rep_Z_O_H_A  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 2], -1), [1 ,1, max_num_neighs_H_A] )
    r_in_gath_Z_O_H_A = tf.gather(inputs[:,:,2], Idx_O_H_A, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_O_H_A = r_in_gath_X_O_H_A - r_in_rep_X_O_H_A
    r_diff_X_O_H_A = r_diff_X_O_H_A - L * tf.round(r_diff_X_O_H_A/L)
    r_diff_Y_O_H_A = r_in_gath_Y_O_H_A - r_in_rep_Y_O_H_A
    r_diff_Y_O_H_A = r_diff_Y_O_H_A - L * tf.round(r_diff_Y_O_H_A/L)
    r_diff_Z_O_H_A = r_in_gath_Z_O_H_A - r_in_rep_Z_O_H_A
    r_diff_Z_O_H_A = r_diff_Z_O_H_A - L * tf.round(r_diff_Z_O_H_A/L)
    norm_O_H_A = tf.sqrt(tf.square(r_diff_X_O_H_A) + tf.square(r_diff_Y_O_H_A) + tf.square(r_diff_Z_O_H_A))
    binv_O_H_A = tf.math.reciprocal(norm_O_H_A)
    sRij_O_H_A = gaussianSwitch(norm_O_H_A, inner_factor_A * radious_A, radious_A) # 未改变顺序之前 
    bx_O_H_A = tf.math.multiply(r_diff_X_O_H_A, binv_O_H_A)
    by_O_H_A = tf.math.multiply(r_diff_Y_O_H_A, binv_O_H_A)
    bz_O_H_A = tf.math.multiply(r_diff_Z_O_H_A, binv_O_H_A)    
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_O_H_A)
    sRij_O_H_A = tf.where(mask_O_H_A, sRij_O_H_A, zeroDummy)
    bx_O_H_A = tf.where(mask_O_H_A, bx_O_H_A, zeroDummy)
    by_O_H_A = tf.where(mask_O_H_A, by_O_H_A, zeroDummy)
    bz_O_H_A = tf.where(mask_O_H_A, bz_O_H_A, zeroDummy)
    RI_O_H_A = tf.stack([sRij_O_H_A, sRij_O_H_A * bx_O_H_A, sRij_O_H_A * by_O_H_A, sRij_O_H_A * bz_O_H_A], axis = -1)
    
    # print(RI_O_H_A.shape, sRij_O_H_A.shape)
    # (10, 100, 50, 4) (10, 100, 50)

    r_in_rep_X_H_H_A  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 0], -1), [1 ,1, max_num_neighs_H_A])
    r_in_gath_X_H_H_A = tf.gather(inputs[:,:,0], Idx_H_H_A, batch_dims = 1, axis = 1)
    r_in_rep_Y_H_H_A  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 1], -1), [1 ,1, max_num_neighs_H_A] )
    r_in_gath_Y_H_H_A = tf.gather(inputs[:,:,1], Idx_H_H_A, batch_dims = 1, axis = 1)
    r_in_rep_Z_H_H_A  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 2], -1), [1 ,1, max_num_neighs_H_A] )
    r_in_gath_Z_H_H_A = tf.gather(inputs[:,:,2], Idx_H_H_A, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_H_H_A = r_in_gath_X_H_H_A - r_in_rep_X_H_H_A
    r_diff_X_H_H_A = r_diff_X_H_H_A - L * tf.round(r_diff_X_H_H_A/L)
    r_diff_Y_H_H_A = r_in_gath_Y_H_H_A - r_in_rep_Y_H_H_A
    r_diff_Y_H_H_A = r_diff_Y_H_H_A - L * tf.round(r_diff_Y_H_H_A/L)
    r_diff_Z_H_H_A = r_in_gath_Z_H_H_A - r_in_rep_Z_H_H_A
    r_diff_Z_H_H_A = r_diff_Z_H_H_A - L * tf.round(r_diff_Z_H_H_A/L)
    norm_H_H_A = tf.sqrt(tf.square(r_diff_X_H_H_A) + tf.square(r_diff_Y_H_H_A) + tf.square(r_diff_Z_H_H_A))
    binv_H_H_A = tf.math.reciprocal(norm_H_H_A)
    sRij_H_H_A = gaussianSwitch(norm_H_H_A, inner_factor_A * radious_A, radious_A) # 未改变顺序之前 
    bx_H_H_A = tf.math.multiply(r_diff_X_H_H_A, binv_H_H_A)
    by_H_H_A = tf.math.multiply(r_diff_Y_H_H_A, binv_H_H_A)
    bz_H_H_A = tf.math.multiply(r_diff_Z_H_H_A, binv_H_H_A)    
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_H_H_A)
    sRij_H_H_A = tf.where(mask_H_H_A, sRij_H_H_A, zeroDummy)
    bx_H_H_A = tf.where(mask_H_H_A, bx_H_H_A, zeroDummy)
    by_H_H_A = tf.where(mask_H_H_A, by_H_H_A, zeroDummy)
    bz_H_H_A = tf.where(mask_H_H_A, bz_H_H_A, zeroDummy)
    RI_H_H_A = tf.stack([sRij_H_H_A, sRij_H_H_A * bx_H_H_A, sRij_H_H_A * by_H_H_A, sRij_H_H_A * bz_H_H_A], axis = -1)
    
    # print(RI_H_H_A.shape, sRij_H_H_A.shape)
    # (10, 200, 50, 4) (10, 200, 50)
    # print(r_in_rep_X_O_O_A.shape, r_in_rep_X_H_O_A.shape, r_in_rep_X_O_H_A.shape, r_in_rep_X_H_H_A.shape)
    # (10, 100, 25) (10, 200, 25) (10, 100, 50) (10, 200, 50)
    
    RI_O_A = tf.concat([RI_O_O_A, RI_H_O_A], axis = 1)
    RI_H_A = tf.concat([RI_O_H_A, RI_H_H_A], axis = 1)
    RI_A = tf.concat([RI_O_A, RI_H_A], axis = 2)
    # print(RI_A.shape, "Here") # (10, 300, 75, 4)
    
    #tf.print(RI_O_H_A[0,0])

    ##########################################################################
    # extract per_dimension the repeated and gathered entries
    r_in_rep_X_O_O_R  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 0], -1), [1 ,1, max_num_neighs_O_R])
    r_in_gath_X_O_O_R = tf.gather(inputs[:, :(Npoints_Divide_3), 0], Idx_O_O_R, batch_dims = 1, axis = 1)
    r_in_rep_Y_O_O_R  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 1], -1), [1 ,1, max_num_neighs_O_R] )
    r_in_gath_Y_O_O_R = tf.gather(inputs[:, :(Npoints_Divide_3), 1], Idx_O_O_R, batch_dims = 1, axis = 1)
    r_in_rep_Z_O_O_R  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 2], -1), [1 ,1, max_num_neighs_O_R] )
    r_in_gath_Z_O_O_R = tf.gather(inputs[:, :(Npoints_Divide_3), 2], Idx_O_O_R, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_O_O_R = r_in_gath_X_O_O_R - r_in_rep_X_O_O_R
    r_diff_X_O_O_R = r_diff_X_O_O_R - L * tf.round(r_diff_X_O_O_R/L)
    r_diff_Y_O_O_R = r_in_gath_Y_O_O_R - r_in_rep_Y_O_O_R
    r_diff_Y_O_O_R = r_diff_Y_O_O_R - L * tf.round(r_diff_Y_O_O_R/L)
    r_diff_Z_O_O_R = r_in_gath_Z_O_O_R - r_in_rep_Z_O_O_R
    r_diff_Z_O_O_R = r_diff_Z_O_O_R - L * tf.round(r_diff_Z_O_O_R/L)
    norm_O_O_R = tf.sqrt(tf.square(r_diff_X_O_O_R) + tf.square(r_diff_Y_O_O_R) + tf.square(r_diff_Z_O_O_R))
    sRij_O_O_R = gaussianSwitch(norm_O_O_R, inner_factor_R * radious_R, radious_R) # 未改变顺序之前 
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_O_O_R)
    sRij_O_O_R = tf.where(mask_O_O_R, sRij_O_O_R, zeroDummy)

    # print(sRij_O_O_R.shape)
    # (10, 100, 54)

    r_in_rep_X_H_O_R  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 0], -1), [1 ,1, max_num_neighs_O_R])
    r_in_gath_X_H_O_R = tf.gather(inputs[:,:,0], Idx_H_O_R, batch_dims = 1, axis = 1)
    r_in_rep_Y_H_O_R  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 1], -1), [1 ,1, max_num_neighs_O_R] )
    r_in_gath_Y_H_O_R = tf.gather(inputs[:,:,1], Idx_H_O_R, batch_dims = 1, axis = 1)
    r_in_rep_Z_H_O_R  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 2], -1), [1 ,1, max_num_neighs_O_R] )
    r_in_gath_Z_H_O_R = tf.gather(inputs[:,:,2], Idx_H_O_R, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_H_O_R = r_in_gath_X_H_O_R - r_in_rep_X_H_O_R
    r_diff_X_H_O_R = r_diff_X_H_O_R - L * tf.round(r_diff_X_H_O_R/L)
    r_diff_Y_H_O_R = r_in_gath_Y_H_O_R - r_in_rep_Y_H_O_R
    r_diff_Y_H_O_R = r_diff_Y_H_O_R - L * tf.round(r_diff_Y_H_O_R/L)
    r_diff_Z_H_O_R = r_in_gath_Z_H_O_R - r_in_rep_Z_H_O_R
    r_diff_Z_H_O_R = r_diff_Z_H_O_R - L * tf.round(r_diff_Z_H_O_R/L)
    norm_H_O_R = tf.sqrt(tf.square(r_diff_X_H_O_R) + tf.square(r_diff_Y_H_O_R) + tf.square(r_diff_Z_H_O_R))
    sRij_H_O_R = gaussianSwitch(norm_H_O_R, inner_factor_R * radious_R, radious_R) # 未改变顺序之前   
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_H_O_R)
    sRij_H_O_R = tf.where(mask_H_O_R, sRij_H_O_R, zeroDummy)

    # print(sRij_H_O_R.shape)
    # (10, 200, 54)

    r_in_rep_X_O_H_R  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 0], -1), [1 ,1, max_num_neighs_H_R])
    r_in_gath_X_O_H_R = tf.gather(inputs[:,:,0], Idx_O_H_R, batch_dims = 1, axis = 1)
    r_in_rep_Y_O_H_R  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 1], -1), [1 ,1, max_num_neighs_H_R] )
    r_in_gath_Y_O_H_R = tf.gather(inputs[:,:,1], Idx_O_H_R, batch_dims = 1, axis = 1)
    r_in_rep_Z_O_H_R  = tf.tile(tf.expand_dims(inputs[:, :(Npoints_Divide_3), 2], -1), [1 ,1, max_num_neighs_H_R] )
    r_in_gath_Z_O_H_R = tf.gather(inputs[:,:,2], Idx_O_H_R, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_O_H_R = r_in_gath_X_O_H_R - r_in_rep_X_O_H_R
    r_diff_X_O_H_R = r_diff_X_O_H_R - L * tf.round(r_diff_X_O_H_R/L)
    r_diff_Y_O_H_R = r_in_gath_Y_O_H_R - r_in_rep_Y_O_H_R
    r_diff_Y_O_H_R = r_diff_Y_O_H_R - L * tf.round(r_diff_Y_O_H_R/L)
    r_diff_Z_O_H_R = r_in_gath_Z_O_H_R - r_in_rep_Z_O_H_R
    r_diff_Z_O_H_R = r_diff_Z_O_H_R - L * tf.round(r_diff_Z_O_H_R/L)
    norm_O_H_R = tf.sqrt(tf.square(r_diff_X_O_H_R) + tf.square(r_diff_Y_O_H_R) + tf.square(r_diff_Z_O_H_R))
    sRij_O_H_R = gaussianSwitch(norm_O_H_R, inner_factor_R * radious_R, radious_R) # 未改变顺序之前 
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_O_H_R)
    sRij_O_H_R = tf.where(mask_O_H_R, sRij_O_H_R, zeroDummy)

    # print(sRij_O_H_R.shape)
    # (10, 100, 108)

    r_in_rep_X_H_H_R  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 0], -1), [1 ,1, max_num_neighs_H_R])
    r_in_gath_X_H_H_R = tf.gather(inputs[:,:,0], Idx_H_H_R, batch_dims = 1, axis = 1)
    r_in_rep_Y_H_H_R  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 1], -1), [1 ,1, max_num_neighs_H_R] )
    r_in_gath_Y_H_H_R = tf.gather(inputs[:,:,1], Idx_H_H_R, batch_dims = 1, axis = 1)
    r_in_rep_Z_H_H_R  = tf.tile(tf.expand_dims(inputs[:, (Npoints_Divide_3):, 2], -1), [1 ,1, max_num_neighs_H_R] )
    r_in_gath_Z_H_H_R = tf.gather(inputs[:,:,2], Idx_H_H_R, batch_dims = 1, axis = 1)
    # compute the periodic dimension wise distance
    r_diff_X_H_H_R = r_in_gath_X_H_H_R - r_in_rep_X_H_H_R
    r_diff_X_H_H_R = r_diff_X_H_H_R - L * tf.round(r_diff_X_H_H_R/L)
    r_diff_Y_H_H_R = r_in_gath_Y_H_H_R - r_in_rep_Y_H_H_R
    r_diff_Y_H_H_R = r_diff_Y_H_H_R - L * tf.round(r_diff_Y_H_H_R/L)
    r_diff_Z_H_H_R = r_in_gath_Z_H_H_R - r_in_rep_Z_H_H_R
    r_diff_Z_H_H_R = r_diff_Z_H_H_R - L * tf.round(r_diff_Z_H_H_R/L)
    norm_H_H_R = tf.sqrt(tf.square(r_diff_X_H_H_R) + tf.square(r_diff_Y_H_H_R) + tf.square(r_diff_Z_H_H_R))
    sRij_H_H_R = gaussianSwitch(norm_H_H_R, inner_factor_R * radious_R, radious_R) # 未改变顺序之前 
    # 把不要的近邻部分设置为0
    zeroDummy = tf.zeros_like(norm_H_H_R)
    sRij_H_H_R = tf.where(mask_H_H_R, sRij_H_H_R, zeroDummy)
    
    #print(sRij_H_H_R.shape)
    # (10, 200, 108)
    
    return RI_A, tf.expand_dims(sRij_O_O_A,-1), tf.expand_dims(sRij_H_O_A,-1), tf.expand_dims(sRij_O_H_A,-1), tf.expand_dims(sRij_H_H_A,-1), tf.expand_dims(sRij_O_O_R,-1), tf.expand_dims(sRij_H_O_R,-1), tf.expand_dims(sRij_O_H_R,-1), tf.expand_dims(sRij_H_H_R,-1)
