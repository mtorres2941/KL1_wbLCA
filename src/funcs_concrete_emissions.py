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

from funcs_log_tools import (erfi, cval, logparams)



#### THIS FUNCTION IS FROM ConcreteUncertaintyModel - concrete_emissions_m3
#class of fly ash [C, F, N/A], for which CaO concentration is [20-40%, 0-7%, 0-40%], uniform distribution
def concrete_emissions_m3(wt_dict_tonnes, flyash_class='idk', staticlist: list=[]):
    """
    Produce randomized mixes in accordance with concrete code. Input weight just needs to be consistent between materials. Emissions based on DeRousseau et al, 2020
    
    INPUTS:
    wt_dict_tonnes, flyash_class='idk', staticlist: list=[]
    
    OUTPUT:
    df_emissions
    """
    
    #########################
    ### GET WEIGHTS IN KG ###
    #########################
    kgm3_pcy = 0.59327642
    
    #initialize densities
    density_dict_kgm3 = { #https://theconstructor.org/building/density-construction-materials/13531/
        'Cement': 3150,
        # 'Coarse Aggregate, Rodded': 1650,
        'Coarse Aggregate': 2700,
        'Fine Aggregate': 2660, #spec gravity of sand = 2.66 https://www.pmec.ac.in/images/lm/3_GTE_Lab_Manual.pdf
        'Water': 1000,
        'Fly Ash': np.random.uniform(2100, 3000), # The specific gravity of fly ash usually ranges from 2.1 to 3.0 https://www.fhwa.dot.gov/publications/research/infrastructure/structures/97148/cfa51.cfm#:~:text=The%20specific%20gravity%20of%20fly,unburned%20carbon%20in%20the%20ash.
    }


    #Check materials in wt_dict_tonnes
    for key in density_dict_kgm3:
        if key not in wt_dict_tonnes:
            wt_dict_tonnes[key] = 0
    
    for key in wt_dict_tonnes:
        if key not in density_dict_kgm3:
            raise ValueError(f'"{key}" not in density_dict_kgm3, which contains {list(density_dict_kgm3.keys())}')


    #calculate volume in m3
    vol_dict_m3 = dict.fromkeys(wt_dict_tonnes.keys())
    vol_m3 = 0
    for key in wt_dict_tonnes:
        vol = wt_dict_tonnes[key]/density_dict_kgm3[key]*1000
        vol_m3 += vol
        vol_dict_m3[key] = vol
    
    
    ###########
    #INITIALIZING ALL OF THE VARIABLES AS SHOWN IN: DeRousseau et al 2020
    ###########
    
    #All emissions calculations assume weight is input in tonnes based on the original paper. This is corrected for at the end of the function
    
    #A1 RAW MATERIAL SUPPLY
    cem_lo = 0.325*2.20462*1000 #originally per lb according to EC3 on 2023-11-17, converted to per tonne
    cem_hi = 0.4277*2.20462*1000 

    if 'cement_a1' in staticlist:
        cement_a1 = np.mean([cem_lo, cem_hi])
    else:
        cement_a1 = np.random.uniform(cem_lo, cem_hi, 1)[0]
    
    if 'fineagg_a1' in staticlist:
        fineagg_a1 = 1.85+37.95 #kg CO2 / tonne fine agg
    else:
        fineagg_a1 = np.random.triangular(0.25, 1.85, 3.45) #kg CO2 / tonne fine agg fuel use for land-won acquisition
        fineagg_a1 += np.random.triangular(34.24, 37.95, 41.65) #kg CO2 / tonne fine agg fuel use for marine dredging
    
    # if 'superplast_a1' in staticlist:
    #     superplast_a1 = 1792 #kg CO2 / tonne
    # else:
    #     superplast_a1 = np.random.normal(loc=1792, scale=428) #kg CO2 / tonne

    #A2 TRANSPORTATION
    #seems incomplete because one distance is given, then emissions for two modes of transportation
    #i'm going to assume everything is transported by truck
    
    #values taken from EPA https://www.epa.gov/sites/default/files/2018-03/documents/emission-factors_mar_2018_0.pdf
    truck_CO2_kmtonne = (0.202 + 0.002e-3*25 + 0.0015e-3*298)*1.10231/1.60934 #kg CO2 / tonne / km (conversions: CH4 and N2O to GWP-100, ton-tonne, mile-km)
    rail_CO2_kmtonne = (0.023 + 0.0018e-3*25 + 0.0006e-3*298)*1.10231/1.60934
    ship_CO2_kmtonne = (0.059 + 0.0005e-3*25 + 0.004e-3*298)*1.10231/1.60934
    air_CO2_kmtonne = (1.308 + 0*25 + 0.0402e-3*293)*1.10231/1.60934
    transp_CO2_kmtonne = [truck_CO2_kmtonne, rail_CO2_kmtonne, ship_CO2_kmtonne, air_CO2_kmtonne]
    
    if 'cement_a2' in staticlist:
        cement_km = [102.5, 0, 0, 0] #distance in km
    else:
        cement_km = [np.random.normal(loc=102.5, scale=48.7), 0, 0, 0]
    
    if 'coarseagg_a2' in staticlist:
        coarseagg_km = [26.1, 0, 0, 0]
    else:
        coarseagg_km = [np.random.normal(loc=26.1, scale=10.5), 0, 0, 0]
    
    if 'fineagg_a2' in staticlist:
        fineagg_km = [25.9, 0, 0, 0]
    else:
        fineagg_km = [np.random.normal(loc=25.9, scale=12), 0, 0, 0]
    
    cement_a2 = np.multiply(cement_km, transp_CO2_kmtonne).sum()
    coarseagg_a2 = np.multiply(coarseagg_km, transp_CO2_kmtonne).sum()
    fineagg_a2 = np.multiply(fineagg_km, transp_CO2_kmtonne).sum()
                                 
    
    #A3 MANUFACTURING
    diesel_CO2_L = 3.152 #kg CO2e/unit fuel (L)
    naturalgas_CO2_m3 = 2.386 #kg CO2e/unit fuel (m3)

    if 'electricity_emissions' in staticlist:
        electricity_CO2_kWh = 0.453
    else:
        electricity_CO2_kWh = np.random.triangular(0.228, 0.453, 0.757) # kg CO2e/kWh
    
    
    if 'diesel_a3' in staticlist:
        diesel_L = 1.968
    else:
        diesel_L = np.random.normal(loc=1.968, scale=0.328) #L diesel / m3 concrete
    
    if 'naturalgas_a3' in staticlist:
        naturalgas_m3 = 0.336
    else:
        naturalgas_m3 = np.random.normal(loc=0.336, scale=0.079) #m3 natural gas /m3 concrete
    
    if 'electricity_a3' in staticlist:
        electricity_kwh = 5.050
    else:
        electricity_kwh = np.random.normal(loc=5.050, scale=0.913) #kWh/m3 concrete
    
    diesel_a3 = diesel_L*diesel_CO2_L
    naturalgas_a3 = naturalgas_m3*naturalgas_CO2_m3
    electricity_a3 = electricity_kwh*electricity_CO2_kWh
    fuel_a3_m3 = diesel_a3 + naturalgas_a3 + electricity_a3 #kg CO2e / m3 concrete

    #B1+C4 CARBONATION = USE/APPLICATION and DISPOSAL
    
    if 'carbonation' in staticlist:
        cem1_CO2_tonneCaO = 610.8 #kg CO2e/tonne CaO
        cem2_CO2_tonneCaO = 681.7
    else:
        cem1_CO2_tonneCaO = np.random.normal(loc=610.8, scale=158) #kg CO2/tonne CaO
        cem2_CO2_tonneCaO = np.random.normal(loc=681.7, scale=139.6)
        
    if 'flyash_class' in staticlist:
        if flyash_class=='C':
            cao_percentage_by_flyash = 0.3# percentage by weight
        elif flyash_class=='F':
            cao_percentage_by_flyash = 0.035
        else:
            cao_percentage_by_flyash = 0.2
    else:
        if flyash_class=='C':
            cao_percentage_by_flyash = np.random.uniform(0.2, 0.4) # percentage by weight
        elif flyash_class=='F':
            cao_percentage_by_flyash = np.random.uniform(0.01, 0.07)
        else:
            cao_percentage_by_flyash = np.random.triangular(0.01, .2, 0.4)
            
    uptake_CO2_tonnecement = 0.49*1000 # maximum uptake of CO2 per tonne of portland cement

    carbonation =  cao_percentage_by_flyash*wt_dict_tonnes['Fly Ash']*(cem1_CO2_tonneCaO+cem2_CO2_tonneCaO) + uptake_CO2_tonnecement*wt_dict_tonnes['Cement'] #total carbonation is in kg CO2e
        
    
    ###########
    # GETTING ACTUAL EMISSIONS
    ###########
    
    #create dataframe of emissions. Index is material, column is stage 
    columns=['A1', 'A2', 'A3', 'B1+C4']
    emission_factors_tonne={
        'Cement': [cement_a1, cement_a2, 0, 0],
        # 'Coarse Aggregate, Rodded': [0, 0, 0, 0],
        'Coarse Aggregate': [0, coarseagg_a2, 0, 0],
        'Fine Aggregate': [fineagg_a1, fineagg_a2, 0, 0],
        'Fly Ash': [0, 0, 0, 0], #fly ash determines carbonation but is a byproduct so it doesn't have emissions of its own
        'Water': [0, 0, 0, 0],
        'Concrete': [0, 0, fuel_a3_m3, -1*carbonation/vol_m3],
    }

    df_emissions = pd.DataFrame.from_dict(data=emission_factors_tonne, orient='index', columns=columns)
    df_emissions = df_emissions.T

    for mat in wt_dict_tonnes.keys():
        df_emissions[mat] = df_emissions[mat]*wt_dict_tonnes[mat]/vol_m3
    df_emissions = df_emissions.T
    
    return df_emissions