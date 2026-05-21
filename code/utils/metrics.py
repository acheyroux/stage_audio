import numpy as np

def signal_power(x):
    return np.sum(x**2)/len(x)

def gSNR(u,v):
    Pu,Pv=np.sum(u**2)/len(u),np.sum(v**2)/len(v)
    return 10*np.log10(Pu/Pv)

def SNR(x,xd):
    '''
    Renvoie le SNR pour le signal pur et debruite
    --- In ---
    x : signal pur (ndarray)
    xd : signal debruite (ndarray)
    --- Out ---
    None
    '''
    max_len=min(len(x),len(xd))
    Px,Pxd=np.sum(x[:max_len]**2)/max_len,np.sum((xd[:max_len]-x[:max_len])**2)/max_len
    return 10*np.log10(Px/Pxd)