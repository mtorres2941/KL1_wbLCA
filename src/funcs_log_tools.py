import pandas as pd
import numpy as np
from numpy import log as ln

import math
pi = math.pi
e = math.e
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones, cov)
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.graph_objects as go
from scipy import (stats, linalg, special)
from scipy.stats import (norm, lognorm, gamma, gaussian_kde, rv_continuous)
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import (GridSearchCV, LeaveOneOut)
import seaborn as sns
from textwrap import wrap
from matplotlib import (pylab, rc, transforms)

import openturns as ot
from itertools import combinations
from colorama import Fore, Style
import rich
import datetime



#THIS FUNCTION IS FROM LogTool - logparams
def erfi(x):
    """
    This is the error function.
    """
    a = 8*(pi-3)/(3*pi*(4-pi));
    xinv = sqrt(-2/(pi*a) - ln(1-x**2)/2 + sqrt((2/(pi*a) + ln(1-x**2)/2)**2 - ln(1-x**2)/a))
    return xinv

def cval(p):
    """
    Use the error function to calculate the value of c
    """
    if p<=0.0 or 1.0<=p:
        raise ValueError("p must be between 0 and 1")
    elif p<0.5:
        c = -sqrt(2)*erfi(2*p-1)
    elif p>=0.5:
        c = sqrt(2)*erfi(2*p-1)
    return c


def logparams(xlow, xmle, xhigh, plow, phigh, tolerance=1e-6):
    """
    xlow is the lower value we have for the distribution. plow is the percentile of this value
    xhigh and phigh are the higher value and respective percentile
    xmle is the most likely estimate
    lognormal distribution is skewed right, so xhigh must be equidistant from xlow and xhigh or closer to xlow
    """

    #check that variables are acceptable
    if plow<=0.0 or 1.0<=plow:
        raise ValueError("plow must be between 0 and 1")
    elif phigh<=0.0 or 1.0<=phigh:
        raise ValueError("phigh must be between 0 and 1")
    elif xlow <=0:
        raise ValueError("xlow must be positive")
    elif xmle <=0:
        raise ValueError("xmle must be positive")
    elif xhigh <=0:
        raise ValueError("xhigh must be positive")
    elif xmle <= xlow or xhigh <= xmle:
        raise ValueError("xmle must be between xlow and xhigh")
    
    #define our c constants for low and high
    clow = cval(plow)
    chigh = cval(phigh)
    
    #goalseek to find sigma (shape)
    testdiff = tolerance
    lowguess = 0
    highguess = abs(clow)

    guess = (lowguess + highguess)/2

    R = (xhigh-xmle)/(xmle-xlow)    
    testdiff = R - (e**(chigh*guess)-e**(-guess**2))/(e**(-guess**2)-e**(clow*guess))

    while abs(testdiff)>=tolerance:
        if testdiff < 0: #guess is too high
            highguess = guess
            guess = (lowguess + highguess)/2
        elif testdiff > 0: #guess is too low
            lowguess = guess
            guess = (lowguess + highguess)/2
        testdiff = R - (e**(chigh*guess)-e**(-guess**2))/(e**(-guess**2)-e**(clow*guess))
    shape = guess
    
    #use shape to find the location parameter
    loc = (xmle*e**(shape*(clow+shape))-xlow)/(e**(shape*(clow+shape))-1)
    
    #use shape and loc to find scale
    scale = e**shape**2*(xmle-loc)
    
    return shape, loc, scale #shape is sigma, location is theta, and scale is m