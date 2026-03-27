#from funcs_kde import positive_kde_func, find_bandwidth, kernel_func, kde_mass_ratios, kde_building

import pandas as pd
import numpy as np
from numpy import log as ln
from numpy import sqrt as sqrt
import math
pi = math.pi
e = math.e
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones, cov)
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly
import plotly.graph_objects as go
from scipy import (stats, linalg, special)
from scipy.stats import (norm, lognorm, gamma, gaussian_kde, rv_continuous)
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import (GridSearchCV, LeaveOneOut)
import seaborn as sns
import textwrap
from textwrap import wrap
from matplotlib import (pylab, rc, transforms)

import openturns as ot
from itertools import (combinations, permutations, product)
from colorama import Fore, Style
import rich
import datetime
import time
import string


################################################################################################################
################################################################################################################

class positive_kde_func:
    def __init__(self, kde_func):
        self.kde_func = kde_func
        self.pos_area = kde_func.integrate_box_1d(0, float('inf'))
        self.neg_area = 1-self.pos_area
        self.d = kde_func.d
        self.n = kde_func.n
        self.dataset = kde_func.dataset
        self.factor = kde_func.factor
        self.covariance = kde_func.covariance
        self.inv_cov = kde_func.inv_cov

        
    def covariance_factor(self):
        print('the covariance_factor function has not been edited to exclude negative values')
        return self.kde_func.covariance_factor()
    
    def evaluate(self, points):
        
        """https://github.com/scipy/scipy/blob/main/scipy/stats/_kde.py
        Evaluate the estimated pdf on a set of points.
        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.
        Returns
        -------
        values : (# of points,)-array
            The values at each point.
        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.
        """
        points = atleast_2d(asarray(points))[0]
        pdf_list = []
        for x in points:
            if x <= 0:
                pdf = 0
            else:
                pdf = self.kde_func.pdf(x)[0]/self.pos_area
            
            pdf_list.append(pdf)
        
        pdf_array = atleast_2d(asarray(pdf_list))[0]
        return pdf_array
    
    def integrate_gaussian(self, mean, cov):
        print('the integrate_gaussian function has not been edited to exclude negative values')
        return self.kde_func.integrate_gaussian(mean, cov)
    
    def integrate_box_1d(self, low, high):
        area_unprocessed = self.kde_func.integrate_box_1d(low, high)
        area_actual = area_unprocessed/self.pos_area
        return area_actual
    
    def integrate_box(self, low_bounds, high_bounds, maxpts):
        print('the integrate_box function has not been edited to exclude negative values')
        return self.kde_func.integrate_box(low_bounds, high_bounds, maxpts)
    
    def integrate_kde(self, other):
        print('the integrate_kde function has not been edited to exclude negative values')
        return self.kde_func.integrate_kde(other)
    
    def logpdf(self, points):
        values_unprocessed = self.evaluate(points)
        values_list = []
        for val in values_unprocessed:
            val_processed = math.log(val)
            values_list.append(val_processed)
        
        values_processed = atleast_2d(asarray(values_list))[0]
        return values_processed
        
    def pdf(self, x):
        return self.evaluate(x)
    
    def resample(self, size=None, seed=None):
        #this version does not consider random seeds
        if size is None:
            size = int(self.n)
        samples = []
        n = 0
        while n < size:
            x = self.kde_func.resample(size=1)[0][0]
            if x > 0:
                samples.append(x)
                n += 1
        samples = atleast_2d(asarray(samples))
        return samples
    
    def set_bandwidth(self, bw_method):
        return self.kde_func.set_bandwidth(bw_method)
    

    
    
    # custom_func.covariance_factor()    computes coefficient (kde.factor) that multiplies data covariance matrix to obtain kernel covariance matrix
    # custom_func.evaluate(points)    evaluate pdf at point(s)
    # custom_func.integrate_gaussian(mean, cov)    multiply estimated density by multivariate gaussian and integrate over whole space
    # custom_func.integrate_box_1d(low, high)    compute integral of 1D pdf between two bounds
    # custom_func.integrate_box(low_bounds, high_bounds, [maxpts])    computes integral of pdf over rectangular interval
    # custom_func.integrate_kde(other)    compute integral of product of this kde with another
    
    # custom_func.logpdf(points)    evaluate log of estimated pdf on set of points
    # custom_func.pdf(points)    same as evaluate
    
    # custom_func.resample([size, seed])    randomly select dataset from estimated pdf
    # custom_func.set_bandwidth([bw_method])    compute estimator bandwidth with given method


################################################################################################################
################################################################################################################
#only works for 1-dimensional systems
def find_bandwidth(data, bw_method='silverman'):
    """
    return: bandwidth
    
    
    In simple cases (e.g., in the case of unimodal distribution), 
    you can safely use the Scott’s or Silverman’s rule of thumb 
    (the Silverman’s rule is recommended because it’s more robust)
    They work extremely fast and produce an excellent bandwidth value 
    for the normal distribution and distributions close to normal.
    However, if you are not sure about the form of your distribution, 
    you may need a non-parametric bandwidth selector
    
    A reliable data-based bandwidth selection method for kernel density estimation., S. J. Sheather and 
    M. C. Jones, Journal of the Royal Statistical Society. Series B (Methodological), 53(3) :683–690, 1991.

    https://aakinshin.net/posts/kde-bw/    
    https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.KernelSmoothing.html
    """
    data = np.array(data).flatten()
    n = len(data)
    std = np.std(data)
    iqr = np.quantile(data,0.75) - np.quantile(data,0.25)

    if bw_method=='scott':
        bandwidth = 1.06*std*n**(-1/5)
    
    elif bw_method=='silverman': #narrower bandwidth than scott
        n = len(data)
        bandwidth = 0.9*min(std, iqr/1.35)*n**(-1/5)
    
    elif bw_method=='sheather-jones':
        # the time it takes to run this method scales up quadratically
        # it takes about 6 seconds for 10k samples, so don't run it for more than 10k samples
        if len(data) > 10_000:
            raise ValueError('Do not use the sheather-jones method for sample sizes >10k. 10k takes about 6 seconds to run and it scales up quadratically')
        kernel=ot.Normal()
        binned=False
        ks = ot.KernelSmoothing()
        X = np.array(data)[:, np.newaxis]
        bandwidth = ks.computePluginBandwidth(X)[0]
    
    elif bw_method is float or bw_method is int:
        bandwidth = bw_method
    
    else:
        raise ValueError("bw_method must be , 'scott', 'silverman', 'sheather-jones', 'float', or 'int'")
        
    return bandwidth

################################################################################################################
################################################################################################################

def kernel_func(data: list, bw_method='silverman'):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    
    ### ATTRIBUTES ###
    custom_func.dataset    returns original data set
    custom_func.d    returns # dimensions
    custom_func.n    returns # data points
    custom_func.neff    returns effective # data points
    custom_func.factor    returns bandwidth factor
    custom_func.covariance    returns covariance matrix of dataset, scaled by calculated bandwidth
    custom_func.inv_cov    returns inverse of covariance
    custom_func.resample(size=n)    returns array of samples of length 1 with n elements in the first element
    custom_func.resample(size=n)[0]    returns array of samples of length n
    
    
    ### METHODS ###
    custom_func.evaluate(points)    evaluate pdf at point(s)
    custom_func.integrate_gaussian(mean, cov)    multiply estimated density by multivariate gaussian and integrate over whole space
    custom_func.integrate_box_1d(low, high)    compute integral of 1D pdf between two bounds
    custom_func.integrate_box(low_bounds, high_bounds, [maxpts])    computes integral of pdf over rectangular interval
    custom_func.integrate_kde(other)    compute integral of product of this kde with another
    custom_func.pdf(points)    same as evaluate
    custom_func.logpdf(points)    evaluate log of estimated pdf on set of points
    custom_func.resample([size, seed])    randomly select dataset from estimated pdf
    custom_func.set_bandwidth([bw_method])    compute estimator bandwidth with given method
    custom_func.covariance_factor()    computes coefficient (kde.factor) that multiplies data covariance matrix to obtain kernel covariance matrix
    """
    data = np.array(data).flatten()
    bandwidth = find_bandwidth(data, bw_method=bw_method)/np.std(data)
    custom_func = stats.gaussian_kde(dataset=data, bw_method=bandwidth, weights=None) 
    
    return custom_func



################################################################################################################
################################################################################################################

def kde_mass_ratios(mass_ratio_dict, kde_dict, mass_or_vol='mass', mc_runs=1, bw_method='silverman'): #gwp_material, gwp_assembly
    """
    return results_dict, df_mc
    """
    materials = mass_ratio_dict.keys()
    #check to make sure all input materials are in the provided kde_dict
    for material in materials:
        if material not in kde_dict.keys():
            raise ValueError(f"{material} isn't in the kde_dict provided. Check spelling.")
    
    #get place holder material quantities. the totals don't matter, just the proportions
    #this function is turning it into a mass ratio if vol is specified
    if mass_or_vol == 'mass':
        pass
#        for material in materials:
#            mass_ratio_dict[material] = mass_ratio_dict[material]/kde_dict[material]['density_kgm3']
    elif mass_or_vol == 'vol':
        for material in materials:
            mass_ratio_dict[material] = mass_ratio_dict[material]*kde_dict[material]['density_kgm3']
    
    else:
        raise ValueError("mass_or_vol should be unspecified (mass) or specified as 'mass' or 'vol'")
    
    #create dataframe with n=mc_runs iterations for each material
    #find "typical" ecc for each material so we can reference for sensitivity analysis
    
    ## The columns of df_mc will be a n=mc_runs Monte Carlo simulation for each material, 
    ## so one column per material, then the sum of those materials, then the total when each of 
    ## those materials is the only one NOT held constant, then the gap proportion of the first material, which is the material of interest.
    
    ## mle_dict has the median for each material, so this is the value used for each material when it's held constant
    
    df_mc = pd.DataFrame()
    mle_dict = {}
    for material in materials:
        declared_unit = kde_dict[material]['declared_unit']
        if declared_unit == 'kg':
            factor = 1

        elif declared_unit == 'm3':
            factor = 1/kde_dict[material]['density_kgm3']
        
        elif declared_unit == 'm2rsi':
            factor = 1

        else:
            raise ValueError(f"{material} is not coming in with kg, m3, or m2rsi as the unit, so we can't use this method. Convert to these units.")

        #create n=mc_runs of the ECC * mass ratio (kg or m2rsi) * factor (inverse density if vol is specified) to get total kg CO2e
        df_mc[material] = kde_dict[material]['kde_function'][bw_method].resample(size=mc_runs)[0]*factor*mass_ratio_dict[material]
        mle_dict[material] = np.median(kde_dict[material]['kde_function'][bw_method].dataset)*factor*mass_ratio_dict[material]
    
    
    ## First-order sensitivity index is Si = Vi/V(Y), where only one material is 
    # varied at a time and then (supposedly) averaged across different fixed values of the othermaterials
    ## Total-effect index is STi = E_Xi(Var_Xi(Y|X~i))/Var(Y) = 1 - Var_X~i(E_Xi(Y|X~i))/Var(Y)
    ## Martin's variation given an additive independent system: UIi = 1 - Var(Y|X~i)/Var(Y)
    df_mc['pLCA'] = df_mc.sum(axis=1)
    mle_total = sum(list(mle_dict.values()))
    results_dict = {}
    for material in materials:
        string = f'{material} fixed run'
        df_mc[string] = df_mc['pLCA'] - df_mc[material] + mle_dict[material] #keeping one variable fixed
        #df_mc[string] = mle_total - mle_dict[material] + df_mc[material] #varying one variable

        results_dict[material] = {'mean':np.mean(df_mc[string]),
                                  'median':np.median(df_mc[string]),
                                  'var':np.var(df_mc[string]),
                                  'std':np.std(df_mc[string]),
                                 }
    results_dict['pLCA'] = {'mean':np.mean(df_mc['pLCA']),
                            'median':np.median(df_mc['pLCA']),
                            'var':np.var(df_mc['pLCA']),
                            'std':np.std(df_mc['pLCA']),
                           }


    
    mat1 = list(materials)[0]
    df_mc['mat1_gwp_proportion'] = df_mc[mat1]/df_mc['pLCA']
    return results_dict, df_mc


################################################################################################################
################################################################################################################

def kde_building(df_qtys, kde_dict: dict, staticlist: list=[], mc_runs=1, bw_method = 'silverman'): #df_qtys, gwp_material, gwp_assembly
    """
    OUTPUT: results_dict, df_mc keeping one variable fixed   
    """
    
    materials = df_qtys['Material'].unique()
    df_mc = pd.DataFrame(columns=materials)

    #check that each material has one consistent unit used
    units = {}
    for material in materials:
        unit_array = df_qtys.loc[df_qtys['Material']==material]['Unit'].unique()
        if len(unit_array) > 1:
            raise ValueError(f'{material} has multiple units in df_qtys: {units}')
        else:
            units[material] = unit_array[0]

    #check that units match in kde_dict and units
    for material in materials:
        if kde_dict[material]['declared_unit'] != units[material]:
            raise ValueError(f"{material} has mismatched units. kde_dict:{kde_dict[material]['declared_unit']}, units:{units[material]}")

    #initialize the dict with quantities and total EC for each material
    df_mc = pd.DataFrame()
    mle_dict = {}
    qty_dict = {}
    for material in materials:
        #create n=mc_runs of the ECC * qty
        qty_dict[material] = df_qtys[df_qtys['Material']==material]['Qty'].sum()
        
        if bw_method == 'sheather-jones' and 'sheather-jones' not in kde_dict[material]['kde_function']:
            mle_dict[material] = np.median(kde_dict[material]['kde_function']['scott'].dataset)*qty_dict[material]
            if material in staticlist:
                df_mc[material] = mle_dict[material]
            else:
                df_mc[material] = kde_dict[material]['kde_function']['scott'].resample(size=mc_runs)[0]*qty_dict[material]
        else:
            mle_dict[material] = np.median(kde_dict[material]['kde_function'][bw_method].dataset)*qty_dict[material]
            if material in staticlist:
                df_mc[material] = mle_dict[material]
            else:
                df_mc[material] = kde_dict[material]['kde_function'][bw_method].resample(size=mc_runs)[0]*qty_dict[material]

    df_mc['pLCA'] = df_mc.sum(axis=1)

    #create the fixed runs
    results_dict = dict.fromkeys(materials)
    for material in materials:
        string = f'{material} fixed run'
        df_mc[string] = df_mc['pLCA'] - df_mc[material] + mle_dict[material] #keeping one variable fixed
        #df_mc[string] = mle_total - mle_dict[material] + df_mc[material] #varying one variable

        results_dict[material] = {'mean':np.mean(df_mc[string]),
                                  'median':np.median(df_mc[string]),
                                  'var':np.var(df_mc[string]),
                                  'std':np.std(df_mc[string]),
                                 }
    results_dict['pLCA'] = {'mean':np.mean(df_mc['pLCA']),
                            'median':np.median(df_mc['pLCA']),
                            'var':np.var(df_mc['pLCA']),
                            'std':np.std(df_mc['pLCA']),
                           }
    return results_dict, df_mc