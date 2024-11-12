#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as tc
tc.set_default_tensor_type(tc.DoubleTensor)


# In[2]:


from __future__ import division
import sys, platform, os

import numpy as np
import scipy as sp
from scipy.optimize import root

import MyHyperSphericaljl as jl
import MyOtherFunctions as fun


# In[3]:


import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))


# In[4]:


h = 0.673
ns = 0.966
As = 2e-9
c_light = 2.88792458e5
OmegaK = 0
KK = -(h*100/c_light)**2 * OmegaK
Kscale = np.sqrt(np.abs(KK))
l_max = 2000

print('The sign of spacial curvature is 0 (flat universe)')
print('The scale of K is ', Kscale, '/ Mpc')


# In[16]:


pars = camb.set_params(H0=100*h, ombh2=0.0223828, omch2=0.1201075, mnu=6.451439e-04, omk=OmegaK, tau=0.05430842, As=As, ns=ns, halofit_version='mead', lmax=l_max)

einstein_solu_data = camb.CAMBdata()
einstein_solu_data.set_params(pars)
einstein_solu_data.calc_background(pars)


# In[17]:


klist = np.hstack([10**np.linspace(-5, -2, 1500, endpoint=False), np.arange(1e-2, 8e-1, 1e-4)])
# klist = np.hstack([10**np.linspace(-5, -3, 1000, endpoint=False), np.arange(1e-3, 8e-1, 1e-4), np.arange(8e-1, 1.3, 5e-4)])

eta_in = einstein_solu_data.conformal_time(1e8)
eta_0 = einstein_solu_data.conformal_time(0)

# Xlist = np.linspace(np.log10(eta_in), np.log10(eta_0), 1000, endpoint=True)
# start at np.log10(eta_in) is not enough for super-horizon modes
lgetalist = np.linspace(-3.5, np.log10(eta_0), 1000, endpoint=True)
etalist = 10**lgetalist

binlist = np.ceil(np.where(eta_0 * klist / (2*tc.pi) * 5 <1000, 1000, eta_0 * klist / (2*tc.pi) * 5))

print('The number of modes is', len(klist))
print('The smallest length scale is ', eta_in, 'Mpc (initial condition)')
print('The largest length scale is ', eta_0, 'Mpc (nowadays)')


# In[8]:


res = tc.empty([len(klist), l_max])
# HyperSphericaljl = tc.empty([len(klist), 2, len(etalist)])

for itr in range(len(klist)):
    ki = klist[itr]
    
    print(' Solving ki =', ki, '/ Mpc , with i =', itr, '    ', end='\r')
    
    #########################################################
    # 1. Computing the solutions to Einstein Eqns

    Weylk = einstein_solu_data.get_time_evolution(q=ki, eta=etalist, vars=['Weyl'], frame='Newtonian')[:,0]
    
    index0 = (Weylk.nonzero())[0][0]
    Psi_normalized = Weylk
    Psi_normalized[:index0] = 1 
    Psi_normalized[index0:] = Weylk[index0:] / Weylk[index0]

    Psi_normalized = tc.tensor(Psi_normalized)    
    Psi_prime = fun.Compute_direvative(Psi_normalized, lgetalist[-1], lgetalist[0])
    dPsi_dx = Psi_prime / tc.tensor(etalist) / np.log(10)

    # plots[itr, 0] = Psi_normalized
    # plots[itr, 1] = Psi_prime

    #########################################################
    # 2. Computing Hyperspherical Bessel Function with lmax

    NX = np.array(2 * binlist[itr], dtype='int32')
    Xlist = np.hstack([10**np.linspace(np.log10(eta_in), np.log10(eta_0 * 100 / NX), 1000, endpoint=False), 
                        np.linspace(eta_0 * 100 / NX, eta_0, NX - 1, endpoint=True)])
    Xlist = tc.tensor(Xlist)
        
    HyperSphericaljl = jl.Recurrence_specified(ki * (eta_0 - Xlist[:-1]), lmax = l_max)

    #########################################################
    # 3. Integration along line of sight

    dPsi_dx_itp = tc.tensor(np.interp(Xlist, etalist, dPsi_dx))
    FX = dPsi_dx_itp * HyperSphericaljl

    integration_value = fun.My_Integral(Xlist, FX)
    res[itr] = integration_value**2 / ki
    


# In[9]:


np.save('Tlsquare_K0_l2000.npy', np.array(res))


# In[ ]:




