import torch as tc
tc.set_default_tensor_type(tc.DoubleTensor)

# K = 1

def TurningPoint(nu, l):
    return tc.arcsin(tc.sqrt(l*(l+1) / nu**2))

# Definitions for sink and tank functions

def sink(x):
    return tc.sin(x)

def tank(x):
    return tc.tan(x)

def dsink(x):
    return tc.cos(x)

def dtank(x):
    return 1 / tc.cos(x)**2

# Definitions for l=0 and l=1 modes

def Phi0nu(nu, x):
    return tc.sin(nu * x)/ nu / sink(x)

def Phi1nu(nu, x):
    return Phi0nu(nu, x) * (1 / tank(x) - nu / tc.tan(nu * x)) / tc.sqrt(nu**2 - 1)

def dPhi0nu(nu, x):
    return (nu * tc.cos(nu * x) * sink(x) - tc.sin(nu * x) * dsink(x)) / nu / sink(x)**2

def dPhiByPhi_recurrence(nu, l, ratio, x):
    return l / tank(x) + tc.sqrt(nu**2 - (l+1)**2) * ratio


# Recurrence coefficients

def alphal(nu, l, x):
    return (2*l + 1) / tank(x) / tc.sqrt(nu**2 - (l+1)**2)

def betal(nu, l):
    return -tc.sqrt(nu**2 - l**2) / tc.sqrt(nu**2 - (l+1)**2)

def ratio_list(nu_int, x):
    '''
        ratio_l := -y_{l+1} / y_l = beta_{l+1}/(alpha_{l+1} + ...)
                0<=l<=nu-1
        ratio_{nu - 1} = 0
        ratio_{nu - 2} = -y_{nu-1} / y_{nu-2} 
                        = limit_{l->nu-1} (beta_l / alpha_l)
    '''

    res = tc.zeros([nu_int, len(x)])
    nu = tc.tensor(nu_int, dtype=tc.float64)
    
    res[-1] = 0
    res[-2] = - tank(x) / tc.sqrt(2*nu - 1)
    llist = tc.arange(nu_int - 3, 0, -1)
    for li in llist:
        res[li] = betal(nu, li + 1) / (alphal(nu, li + 1, x) + res[li+1])

    return res

def ratio_list_modified(nu_int, lmax, x):
    '''
        ratio_l := -y_{l+1} / y_l = beta_{l+1}/(alpha_{l+1} + ...)
                0<=l<=lmax-1<=nu-1
        ratio_{nu - 1} = 0
        ratio_{nu - 2} = -y_{nu-1} / y_{nu-2} 
                        = limit_{l->nu-1} (beta_l / alpha_l)
    '''
    if nu_int <= lmax: 
        res = tc.zeros([nu_int, len(x)])
        nu = tc.tensor(nu_int, dtype=tc.float64)
        
        res[-1] = 0
        res[-2] = - tank(x) / tc.sqrt(2*nu - 1)
        llist = tc.arange(nu_int - 3, 0, -1)
        for li in llist:
            res[li] = betal(nu, li + 1) / (alphal(nu, li + 1, x) + res[li+1])
        
    else:
        res = tc.zeros([lmax, len(x)])
        nu = tc.tensor(nu_int, dtype=tc.float64)
        
        temp = - tank(x) / tc.sqrt(2*nu - 1)
        for ll in tc.arange(nu_int - 3, lmax -2, -1):
            temp = betal(nu, ll + 1) / (alphal(nu, ll + 1, x) + temp)
        
        res[-1] = temp # which corresponds to y_{lmax}/y_{lmax-1}
        llist = tc.arange(lmax - 2, 0, -1)
        for li in llist:
            res[li] = betal(nu, li + 1) / (alphal(nu, li + 1, x) + res[li+1])

    return res

# Computation

def Recurrence(nu_int, x, lmax = False):
    '''
        Input:  nu          int
                x           (N) torch tensor

        Output: Phi_l^nu    (lmax * N) torch tensor with 0<=l<=nu-1
    '''
    if lmax == False:
        lmax = nu_int

    phi = tc.zeros([lmax, len(x)])
    # dphi = tc.zeros([lmax, len(x)])
    nu = tc.tensor(nu_int, dtype=tc.float64)

    phi[0] = Phi0nu(nu, x)
    phi[1] = Phi1nu(nu, x)

    llist = tc.arange(2, min(nu_int,lmax))
    rlist = ratio_list_modified(nu_int, lmax, x)

    # dphi[0] = dPhi0nu(nu, x)
    # dphi[1] = dPhiByPhi_recurrence(nu, 1, rlist[0], x) * phi[1]

    for li in llist:
        phi[li] = tc.where(x > TurningPoint(nu_int, li),  # turning point condition
                            alphal(nu, li-1, x) * phi[li-1] + betal(nu, li-1) * phi[li-2],  # forward recurrence
                            -rlist[li - 1] * phi[li - 1]) # backward recurrence
        
        # dphi[li] = dPhiByPhi_recurrence(nu, li, rlist[li], x) * phi[li]

    return phi #, dphi

def Recurrence_specified(nu_int, x, lmax = False):
    '''
        Input:  nu          int
                x           (N) torch tensor

        Output: Phi_l^nu    (lmax * N) torch tensor with 0<=l<=nu-1
    '''
    if lmax == False:
        lmax = nu_int

    phi = tc.zeros([lmax, len(x)])
    nu = tc.tensor(nu_int, dtype=tc.float64)

    phi[0] = Phi0nu(nu, x)
    phi[1] = Phi1nu(nu, x)

    llist = tc.arange(2, min(nu_int,lmax))
    rlist = ratio_list_modified(nu_int, lmax, x)

    for li in llist:
        phi[li] = tc.where(x > TurningPoint(nu_int, li),  # turning point condition
                            alphal(nu, li-1, x) * phi[li-1] + betal(nu, li-1) * phi[li-2],  # forward recurrence
                            -rlist[li - 1] * phi[li - 1]) # backward recurrence
    
    phi_at_0 = tc.zeros([lmax, 1])
    phi_at_0[0] = tc.tensor([1.])
    phi_tot = tc.hstack([phi, phi_at_0])

    return phi_tot #, dphi