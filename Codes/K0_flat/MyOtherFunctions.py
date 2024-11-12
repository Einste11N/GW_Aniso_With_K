import torch as tc
tc.set_default_tensor_type(tc.DoubleTensor)

def Compute_direvative(y, xmax, xmin):

    N = len(y)
    h = (xmax - xmin) / (N-1)

    dy = tc.zeros(N)

    dy[0] = -25/12 * y[0] + 4 * y[1] - 3 * y[2] + 4/3 * y[3] - 1/4 * y[4]
    dy[1] = -1/4 * y[0] - 5/6 * y[1] + 3/2 * y[2] - 1/2 * y[3] + 1/12 * y[4]

    dy[2:-2] = 1/12 * y[:-4] - 2/3 * y[1:-3] + 2/3 * y[3:-1] - 1/12 * y[4:]
    
    dy[-2] = 1/4 * y[-1] + 5/6 * y[-2] - 3/2 * y[-3] + 1/2 * y[-4] - 1/12 * y[-5]
    dy[-1] = 25/12 * y[-1] - 4 * y[-2] + 3 * y[-3] - 4/3 * y[-4] + 1/4 * y[-5]

    dy = dy / h

    return dy

def My_Integral(x, y):
    dx = x[1:] - x[:-1]
    ymean = (y[:, 1:] + y[:, :-1]) / 2
    return tc.sum(dx * ymean, dim=(1))