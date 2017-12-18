import numpy as np

def black_body(wk_input,T):
    '''wk is an array of frequencies with units in cm^(-1)
    T is a temperature in Kelvin.

    Returns an array of spectral radiances 
    with units MegaJansky per steradian
    '''
    wk = wk_input*100
    I = 2 * (6.62606896e-34) * (299792458.0)**2 * wk**3  * 1 \
        / (np.exp(((6.62606896e-34)*(299792458.0)*wk \
        / ((1.3806504e-23)*T)))-1)
        
    return I/(2.99792458e-12) ## units