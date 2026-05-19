import numpy as np

def signal_power(x):
    return np.sum(x**2)/len(x)

def SNR(u,v):
    Pu,Pv=np.sum(u**2)/len(u),np.sum(v**2)/len(v)
    return 10*np.log10(Pu/Pv)