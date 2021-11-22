# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:15:43 2021

@author: Niels
"""
import scipy.io
mat = scipy.io.loadmat('Van_der_pol_mu2.mat')
VanDerPolmu2 = mat['data2']


mat = scipy.io.loadmat('Van_der_pol_mu15.mat')
VanDerPolmu15 = mat['data']

from matplotlib import pyplot as plt


plt.figure()
plt.plot(VanDerPolmu2[:,0],VanDerPolmu2[:,1])
plt.figure()
plt.plot(VanDerPolmu2[:,0],VanDerPolmu2[:,2])

plt.figure()
plt.plot(VanDerPolmu15[:,0],VanDerPolmu15[:,1])
plt.figure()
plt.plot(VanDerPolmu15[:,0],VanDerPolmu15[:,2])
