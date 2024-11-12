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
OmegaK = -0.05
KK = -(h*100/c_light)**2 * OmegaK
Kscale = np.sqrt(np.abs(KK))
l_max = 2000

print('The sign of spacial curvature is + (closed universe)')
print('The scale of K is ', Kscale, '/ Mpc')


# In[5]:


pars = camb.set_params(H0=100*h, ombh2=0.0223828, omch2=0.1201075, mnu=6.451439e-04, omk=OmegaK, tau=0.05430842, As=2e-9, ns=0.966, halofit_version='mead', lmax=4000)

einstein_solu_data = camb.CAMBdata()
einstein_solu_data.set_params(pars)
einstein_solu_data.calc_background(pars)


# In[6]:


nulist = np.arange(5, 14000, 1)
# nulist = np.array([100, 1000, 10000])
nulist = np.array(nulist, dtype='int64')
klist = Kscale * np.sqrt(nulist**2 - 1)

eta_in = einstein_solu_data.conformal_time(1e8)
eta_0 = einstein_solu_data.conformal_time(0)

# start at np.log10(eta_in) is not enough for super-horizon modes
lgetalist = np.linspace(-3.5, np.log10(eta_0), 1000, endpoint=True)
etalist = 10**lgetalist

x_in = eta_in * Kscale
x_0 = eta_0 * Kscale
binlist = np.where(nulist<1000, 1000, nulist)

print('The number of modes is', len(klist))
print('The smallest length scale is ', eta_in, 'Mpc (initial condition), corresponding x =', x_in)
print('The largest length scale is ', eta_0, 'Mpc (nowadays), corresponding x =', x_0)


# In[7]:


res = tc.empty([len(klist), l_max])
# plots = tc.empty([len(klist), 2, len(etalist)])

for itr in range(len(klist)):
    nu = nulist[itr]
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
    dPsi_dx = Psi_prime / tc.tensor(etalist) / np.log(10) / Kscale

    # plots[itr, 0] = Psi_normalized
    # plots[itr, 1] = Psi_prime

    #########################################################
    # 2. Computing Hyperspherical Bessel Function with lmax

    NX = 2 * binlist[itr]
    Xlist = Kscale * np.hstack([10**np.linspace(np.log10(eta_in), np.log10(eta_0 * 100 / NX), 1000, endpoint=False), 
                        np.linspace(eta_0 * 100 / NX, eta_0, NX - 1, endpoint=True)])
    Xlist = tc.tensor(Xlist)
    
    HyperSphericaljl = jl.Recurrence_specified(nu, Kscale * eta_0 - Xlist[:-1], lmax = l_max)

    #########################################################
    # 3. Integration along line of sight

    dPsi_dx_itp = tc.tensor(np.interp(Xlist, Kscale * etalist, dPsi_dx))
    FX = dPsi_dx_itp * HyperSphericaljl

    integration_value = fun.My_Integral(Xlist, FX)
    res[itr] = integration_value**2 / nu
    


# In[ ]:


np.save('Tlsquare_K+_l2000.npy', np.array(res))


# In[ ]:



