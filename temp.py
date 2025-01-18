import numpy as np
import cv2
import os
import sys
import torch
import numpy as np
import scipy.io as scio
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from UTILS import IFFT2c, FFT2c
from scipy.io import loadmat
from utils.fastmriBaseUtils import IFFT2c, FFT2c
####################################################
#data = np.load('/data0/chentao/data/LplusSNet/data/20coil/k_cine_multicoil_test.npy')
#csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
#print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
#data = data[100,:,:,:,:]
#csm = csm[100,:,:,:,:] 
#img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #
#print("img:", img.shape)
#
#img_max = np.max(np.abs(img))
#img_norm = np.abs(img) / img_max
#brightness_factor = 3
#img_brightened = np.clip(img_norm * brightness_factor, 0, 1)
#
#def animate(frame):
#    plt.imshow(img_brightened[frame], cmap='gray')  
#    plt.title('Frame {}'.format(frame))
#    plt.axis('off')
#
#anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
#anim.save('test.gif', writer='imagemagick')
###################################################
#data = np.load('/data0/chentao/data/LplusSNet/data/20coil/k_cine_multicoil_test.npy')
#csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
#print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
#data = data[100,:,:,:,:]
#csm = csm[100,:,:,:,:] 
#
#C =loadmat('/data0/huayu/Aluochen/Mypaper5/e_192x18_acs4_R4.mat')
#mask = C['mask'][:]
#mask = np.transpose(mask,[1,0])
#mask = np.expand_dims(mask, axis=1)
#print('mask',mask.shape)
#
#data = data*mask
#img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #
#print("img:", img.shape)
#img_max = np.max(np.abs(img))
#img_norm = np.abs(img) / img_max
#brightness_factor = 3
#img_brightened = np.clip(img_norm * brightness_factor, 0, 1)
#
#def animate(frame):
#    plt.imshow(img_brightened[frame], cmap='gray')  
#    plt.title('Frame {}'.format(frame))
#    plt.axis('off')
#
#anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
#anim.save('testzerofilling.gif', writer='imagemagick')
##########################################

#data = np.load('/data0/huayu/Aluochen/Mypaper5/k-gin_kv/out.npy')
##csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
#print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
#data = data[100:101,:,:,:]
##csm = csm[100,:,:,:,:] 
##img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #
#img = IFFT2c(data)
#img = img[0]
#print("img:", img.shape)
#
#img_max = np.max(np.abs(img))
#img_norm = np.abs(img) / img_max
#brightness_factor = 3
#img_brightened = np.clip(img_norm * brightness_factor, 0, 1)
#
#def animate(frame):
#    plt.imshow(img_brightened[frame], cmap='gray')  
#    plt.title('Frame {}'.format(frame))
#    plt.axis('off')
#
#anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
#anim.save('output_kv.gif', writer='imagemagick')
###################################################


# data = np.load('/data0/chentao/data/LplusSNet/data/20coil/k_cine_multicoil_test.npy')
# csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
# print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[100,:,:,:,:]
# csm = csm[100,:,:,:,:] 

# C =loadmat('/data0/huayu/Aluochen/Mypaper5/e_192x18_acs4_R4.mat')
# mask = C['mask'][:]
# mask = np.transpose(mask,[1,0])
# mask = np.expand_dims(mask, axis=1)
# print('mask',mask.shape)

# data = data*mask
# img = np.expand_dims(np.sum(IFFT2c(data) * np.conj(csm), axis=0), axis=0) #
# print("img:", img.shape)
# ksp = FFT2c(img)

# #for i in range(ksp.shape[3]):
# #    
# #    for j in range(mask.shape[3]):
# ksp[:, 10:15, :, 96:100] = ksp[:, 12:13, :, 96:100]
# img = np.expand_dims(np.sum(ksp[:,10:15,:,:], axis=1), axis=1) / np.expand_dims(np.sum(mask[10:15,:,:], axis=0), axis=0)
# #img = np.expand_dims(np.sum(ksp, axis=1), axis=1) / np.expand_dims(np.sum(mask, axis=0), axis=0)
# #img = np.mean(ksp, axis=1, keepdims=True)
# print("img:", img.shape)
# #img = np.repeat(img, repeats=2, axis=1)

# img = IFFT2c(img)
# img = img[0] 

# img_max = np.max(np.abs(img))
# img_norm = np.abs(img) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# def animate(frame):
#     plt.imshow(img_brightened[frame], cmap='gray')  
#     plt.title('Frame {}'.format(frame))
#     plt.axis('off')

# anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# anim.save('test0F.gif', writer='imagemagick')


#def normal_pdf(length, sensitivity):
#    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)
#
#
#def cartesian_mask(shape, acc, sample_n):
#    """
#    Sampling density estimated from implementation of kt FOCUSS
#    shape: tuple - of form (..., nx, ny)
#    acc: float - doesn't have to be integer 4, 8, etc..
#    """
#    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
#    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
#    lmda = Nx/(2.*acc)
#    n_lines = int(Nx / acc)
#
#    # add uniform distribution
#    pdf_x += lmda * 1./Nx
#
#    if sample_n:
#        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
#        pdf_x /= np.sum(pdf_x)
#        n_lines -= sample_n
#
#    mask = np.zeros((N, Nx))
#    for i in range(N):
#        idx = np.random.choice(Nx, n_lines, False, pdf_x)
#        mask[i, idx] = 1
#
#    if sample_n:
#        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
#
#    size = mask.itemsize
#    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))
#
#    mask = mask.reshape(shape)
#
#    return mask
#
#
def get_cine_mask(acc, acs_lines, x=232, y=256):
   rows = y-acs_lines

   matrix = np.zeros((rows, x))

   ones_per_column = rows//acc #y//acc-acs_lines

   first_column = np.zeros(rows)
   indices = np.linspace(0, rows - 1, ones_per_column, dtype=int)
   first_column[indices] = 1

   for j in range(x):
       matrix[:, j] = np.roll(first_column, j)
       
   insert_rows = np.ones((acs_lines, x))
   new_matrix = np.insert(matrix, rows//2, insert_rows, axis=0)
   print(new_matrix)
#    mask = np.transpose(mask, (0, 2, 1))
#
   mask_datadict = {'mask': np.squeeze(new_matrix)}
##    scio.savemat('random_368x368_mask4x_8line.mat', mask_datadict)  #
   # scio.savemat('/data0/huayu/Aluochen/Mypaper5/e_192x18_acs4_R10.mat', mask_datadict)
   scio.savemat('/nfs/zzy/code/k_gin_base/masks/VISTA/e_192x18_acs4_acs_lines4_R12.mat',mask_datadict)
   print('save success')



def main():
   get_cine_mask(acc=12, acs_lines=4, x=18, y=192)
#
if __name__ == '__main__':
   sys.exit(main())