import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import vegas
from functools import partial 
import WIC_Pheno_Formulae as WIC
from WIC_Pheno_Formulae import *

hbar = 6.58*10**(-16)*10**(-9)   ## GeV*s

#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#
####################################################### VEGAS INTEGRATION #####################################################
#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#

#######################################################
##################   e+e- --> μ1μ1'  ##################
##################    d2σ_dμ1dμ1p    #################
##################     with VEGAS    ##################
#######################################################
def run_MC_dσ_dμdμp(NPoints, sqrtS, μ0, μϼ):
    print('Running dσ/dμdμp MC for √s = ' + str(sqrtS) + ' ... ')
    ############################ Functions ########################## 
    d2σ_dμ1dμ1p_weights = partial(WIC.d2σ_dμ1dμ1p, sqrtS=sqrtS, μ0=μ0, μϼ=μϼ)
    d2σ_dμ1dμ1p_v = partial(WIC.d2σ_dμ1dμ1p_v, sqrtS=sqrtS, μ0=μ0, μϼ=μϼ)
    ########################## Training Vegas ... ################################
    integral = 0.0
    integ = vegas.Integrator([[μ0, sqrtS-μ0], [μ0, sqrtS-μ0]])
#     NPoints_d3σ = 50000
    result = integ(d2σ_dμ1dμ1p_v, nitn=10, neval=NPoints, alpha=0.0005) ############# train !
    result_error = result.sdev 
    result_integral = result.mean
#     print(result_integral, result_error)
    μ1_σ_hist = []; μ1p_σ_hist = [];
    y_histo = []; wgt_array = []; hcube_histo = [];
    for x, y, wgt, hcube in integ.random(yield_hcube=True, yield_y=True):
        wgt_array.append(1/wgt)
        μ1_σ_hist.append(x[0])
        μ1p_σ_hist.append(x[1])
        y_histo.append(y[0])
        hcube_histo.append(hcube)  ## hcube is the *index/label* of the hypercube that the point is in
        integral += wgt * d2σ_dμ1dμ1p_v(x)
    print(np.shape(μ1_σ_hist), np.shape(μ1p_σ_hist))
    ################# Rescaling Weights & Cutting Events Probabilistically ################
    hcube_counter = []
    for i in range(max(hcube_histo)+1):          ## Loops thru hypercubes
        if i%5000==0: print(str(i) + ' of ' + str(max(hcube_histo)+1))
        count = np.sum(np.array(hcube_histo)==i) ## Define count = number of points in hypercube i
        temp = np.full((count), count)           ## Define temp = [count, count, ... ('count' times)]
        hcube_counter = np.append(hcube_counter, temp)
        hcube_counter.flatten()
    prob = np.array(hcube_counter)/(len(μ1_σ_hist))
    raw_weights = np.array(list(map(d2σ_dμ1dμ1p_weights, μ1_σ_hist, μ1p_σ_hist))) / (prob)
    rescaled_raw_weights = raw_weights/max(raw_weights)
    my_random_numbers = np.random.uniform(low=0, high=1, size=len(rescaled_raw_weights))
    rescaled_raw_histo_boolean = rescaled_raw_weights>my_random_numbers
    rescaled_μ1_σ_hist = np.array(μ1_σ_hist)[rescaled_raw_histo_boolean]
    rescaled_μ1p_σ_hist = np.array(μ1p_σ_hist)[rescaled_raw_histo_boolean]
    print('shape of raw hist = ' + str(np.shape(μ1_σ_hist)))
    print('shape of rescaled hist = ' + str(np.shape(rescaled_μ1_σ_hist)))
    print('Unweighting Efficiency(?) = ' + str(np.shape(rescaled_μ1_σ_hist)[0]/np.shape(μ1_σ_hist)[0]))
    #######################################################################################
    print('Returning μ1, μ1p Events, Integral')
    return {"μ1 Events": rescaled_μ1_σ_hist, 
            "μ1p Events": rescaled_μ1p_σ_hist, 
            "Integral":  integral, 
            "All Events": np.transpose(np.array([rescaled_μ1_σ_hist, 
                                                 rescaled_μ1p_σ_hist])),
            "Error":  result_error }


#######################################################
##################   e+e- --> μ1μ1'  ##################
##################      d2σ_cosθ     #################
##################     with VEGAS    ##################
#######################################################
def run_MC_dσ_dcosθ(NPoints):
    print('Running dσ/dcosθ MC ... ')
    ############################ Functions ########################## 
    dσ_dcosθ_weights = partial(WIC.dσ_dcosθ)
    dσ_dcosθ_v = partial(WIC.dσ_dcosθ_v)
    ########################## Training Vegas ... ################################
    integral = 0.0
    integ = vegas.Integrator([[-1, 1]])
    result = integ(dσ_dcosθ_v, nitn=10, neval=NPoints, alpha=0.0005) ############# train !
    result_error = result.sdev 
    result_integral = result.mean
#     print(result_integral, result_error)
    cosθ_σ_hist = []; 
    y_histo = []; wgt_array = []; hcube_histo = [];
    for x, y, wgt, hcube in integ.random(yield_hcube=True, yield_y=True):
        wgt_array.append(1/wgt)
        cosθ_σ_hist.append(x[0])
        y_histo.append(y[0])
        hcube_histo.append(hcube)
        integral += wgt * dσ_dcosθ_v(x)
    print(np.shape(cosθ_σ_hist))
    ################# Rescaling Weights & Cutting Events Probabilistically ################
    hcube_counter = []
    for i in range(max(hcube_histo)+1):          ## Loops thru hypercubes
        if i%5000==0: print(str(i) + ' of ' + str(max(hcube_histo)+1))
        count = np.sum(np.array(hcube_histo)==i) ## Define count = number of points in hypercube i
        temp = np.full((count), count)           ## Define temp = [count, count, ... ('count' times)]
        hcube_counter = np.append(hcube_counter, temp)
        hcube_counter.flatten()
    prob = np.array(hcube_counter)/(len(cosθ_σ_hist))
    raw_weights = np.array(list(map(dσ_dcosθ_weights, cosθ_σ_hist))) / (prob)
    rescaled_raw_weights = raw_weights/max(raw_weights)
    my_random_numbers = np.random.uniform(low=0, high=1, size=len(rescaled_raw_weights))
    rescaled_raw_histo_boolean = rescaled_raw_weights>my_random_numbers
    rescaled_cosθ_σ_hist = np.array(cosθ_σ_hist)[rescaled_raw_histo_boolean]
    print(np.shape(cosθ_σ_hist))
    print(np.shape(rescaled_cosθ_σ_hist))
    #######################################################################################
    print('Returning cosθ Events, Integral')
#     print(integral)
    return {"cosθ Events": rescaled_cosθ_σ_hist, 
            "Integral":  integral, 
            "All Events": np.transpose(np.array([rescaled_cosθ_σ_hist])),
            "Error":  result_error }


#######################################################
##################    μ1 --> Z + μ2  ##################
##################      dΓ2_dμ2      ##################
##################     with VEGAS    ##################
#######################################################
def run_MC_dΓ2_dμ2(NPoints, μ1, μ0, μϼ):
    μ1_temporary = μ1
    print('Running dΓ2/dμ2 MC for μ1 = ' + str(μ1_temporary) + ' ... ')
    ############################ Functions ########################## 
    dΓ2_dμ2_weights = partial(WIC.dΓ2_dμ2, μ1=μ1_temporary, μ0=μ0, μϼ=μϼ)
    dΓ2_dμ2_v = partial(WIC.dΓ2_dμ2_v, μ1=μ1_temporary, μ0=μ0, μϼ=μϼ)
    ########################## Training Vegas ... ################################
    integral_Γ2 = 0.0
    integral_Γ2_piece = 0.0
    integ = vegas.Integrator([[μ0, μ1_temporary-mZ]])
#     NPoints_d2Γ2 = 100000
    integ(dΓ2_dμ2_v, nitn=10, neval=NPoints, alpha=0.0005) ############# train !
    μ2_Γ2_hist = []; y_histo = []; wgt_array = []; hcube_histo = [];
    for x, y, wgt, hcube in integ.random(yield_hcube=True, yield_y=True):
        wgt_array.append(1/wgt)
        μ2_Γ2_hist.append(x[0])
        y_histo.append(y[0])
        hcube_histo.append(hcube)
        integral_Γ2 += wgt * dΓ2_dμ2_v(x)
        if x[0]<μ1_temporary-mZ-3: integral_Γ2_piece += wgt * dΓ2_dμ2_v(x)
    print(integral_Γ2)     
    ############ Rescaling Weights & Cutting Events Probabilistically ###########
    hcube_counter = []
    for i in range(max(hcube_histo)+1):          ## Loops thru hypercubes
        if i%5000==0: print(str(i) + ' of ' + str(max(hcube_histo)+1))
        count = np.sum(np.array(hcube_histo)==i) ## Define count = number of points in hypercube i
        temp = np.full((count), count)           ## Define temp = [count, count, ... ('count' times)]
        hcube_counter = np.append(hcube_counter, temp)
        hcube_counter.flatten()
    prob = np.array(hcube_counter)/(len(μ2_Γ2_hist))
    raw_weights = np.array(list(map(dΓ2_dμ2_weights, μ2_Γ2_hist))) / (prob)
    rescaled_raw_weights = raw_weights/max(raw_weights)
    my_random_numbers = np.random.uniform(low=0, high=1, size=len(rescaled_raw_weights))
    rescaled_raw_histo_boolean = rescaled_raw_weights>my_random_numbers
    rescaled_μ2_Γ2_hist = np.array(μ2_Γ2_hist)[rescaled_raw_histo_boolean]
    # rescaled_cosθ_Γ2_hist = np.array(cosθ_Γ2_hist)[rescaled_raw_histo_boolean]
    print('Returning μ2 Events, Integral')
    return {"μ2 Events": rescaled_μ2_Γ2_hist, "Integral": integral_Γ2}
    #######################################################################
    


## Find Closest Element
def ClosestElement(x, a):
    differences = np.array(a)-x
    AbsDifferences = abs(differences)
    ClosestIndex = np.where(AbsDifferences == min(AbsDifferences))[0][0]
    ClosestEl = a[ClosestIndex]
    return ClosestEl
## Determine Δm
def Δm(m, MassSpec):
    CloseM = ClosestElement(m,MassSpec)
    CloseMIndex = np.where(MassSpec == CloseM)[0][0]
    if CloseMIndex!=0:
        dm = CloseM - MassSpec[CloseMIndex-1]
    else:
        dm = abs(CloseM - MassSpec[CloseMIndex+1])
    return dm
## Determine Δm2
def Δm2(m, MassSpec):
    CloseM = ClosestElement(m,MassSpec)
    CloseMIndex = np.where(MassSpec == CloseM)[0][0]
    MassSqSpec = MassSpec**2
    if CloseMIndex!=0:
        dm2 = MassSqSpec[CloseMIndex] - MassSqSpec[CloseMIndex-1]
#         dm = CloseM - MassSpectrum[CloseMIndex-1]
    else:
#         dm = abs(CloseM - MassSpectrum[CloseMIndex+1])
        dm2 = abs(MassSqSpec[CloseMIndex] - MassSqSpec[CloseMIndex+1])
    return dm2
##########################################################
#################  μ1 --> f3 + f4 + μ2  ##################
#################     d3Γ_dμ2dx3dx4     ##################
#################      with VEGAS       ##################
##########################################################
def run_MC_dΓ3_dμ2(NPoints, μ1, μ0, μϼ, Δm2):
    μ1_temporary = μ1
    print('Running dΓ3/dμ2 MC for μ1 = ' + str(μ1_temporary) + ' ... ')
    ######################################## Integrands ################################## 
    d3Γ3_dμ2dx3dx4_weights = partial(WIC.d3Γ3_dμ2dx3dx4, μ1=μ1_temporary, μ0=μ0, μϼ=μϼ, Δm2=Δm2)
    d3Γ3_dμ2dx3dx4_v = partial(WIC.d3Γ3_dμ2dx3dx4_v, μ1=μ1_temporary, μ0=μ0, μϼ=μϼ, Δm2=Δm2)
    ################################### Training Vegas ... #########################################
    integral_Γ3 = 0.0
    integral_Γ3_piece = 0.0
    integ = vegas.Integrator([[μ0, μ1_temporary], 
                              [0,1-(μ0/μ1_temporary)**2], 
                              [0,1-(μ0/μ1_temporary)**2]])
#     NPoints_d3Γ3 = 100000
    integ(d3Γ3_dμ2dx3dx4_v, nitn=10, neval=NPoints, alpha=0.0005) ############# train !
    μ2_Γ3_hist = []; x3_Γ3_hist = []; x4_Γ3_hist = [];
    y_histo = []; wgt_array = []; hcube_histo = [];
    for x, y, wgt, hcube in integ.random(yield_hcube=True, yield_y=True):
        wgt_array.append(1/wgt)
        μ2_Γ3_hist.append(x[0])
        x3_Γ3_hist.append(x[1])
        x4_Γ3_hist.append(x[2])
        y_histo.append(y[0])
        hcube_histo.append(hcube)
        integral_Γ3 += wgt * d3Γ3_dμ2dx3dx4_v(x)
        if x[0]>μ1_temporary-mZ-3:                           ###
            integral_Γ3_piece += wgt * d3Γ3_dμ2dx3dx4_v(x)   ### Defining the Γ3-piecewise integral 
    print(integral_Γ3)
    print(np.shape(μ2_Γ3_hist), np.shape(x3_Γ3_hist), np.shape(x4_Γ3_hist))
    ####################### Rescaling Weights & Cutting Events Probabilistically #####################
    hcube_counter = []
    for i in range(max(hcube_histo)+1):          ## Loops thru hypercubes
        if i%5000==0: print(str(i) + ' of ' + str(max(hcube_histo)+1))
        count = np.sum(np.array(hcube_histo)==i) ## Define count = number of points in hypercube i
        temp = np.full((count), count)           ## Define temp = [count, count, ... ('count' times)]
        hcube_counter = np.append(hcube_counter, temp)
        hcube_counter.flatten()
    prob = np.array(hcube_counter)/(len(μ2_Γ3_hist))
    raw_weights = np.array(list(map(d3Γ3_dμ2dx3dx4_weights, μ2_Γ3_hist, x3_Γ3_hist, x4_Γ3_hist))) / (prob)
    rescaled_raw_weights = raw_weights/max(raw_weights)
    my_random_numbers = np.random.uniform(low=0, high=1, size=len(rescaled_raw_weights))
    rescaled_raw_histo_boolean = rescaled_raw_weights>my_random_numbers
    rescaled_μ2_Γ3_hist = np.array(μ2_Γ3_hist)[rescaled_raw_histo_boolean]
    rescaled_x3_Γ3_hist = np.array(x3_Γ3_hist)[rescaled_raw_histo_boolean]
    rescaled_x4_Γ3_hist = np.array(x4_Γ3_hist)[rescaled_raw_histo_boolean]
    print('Returning μ2 Events, x3 Events, x4 Events, Integral')
    return {"μ2 Events":  rescaled_μ2_Γ3_hist, 
            "x3 Events":  rescaled_x3_Γ3_hist,
            "x4 Events":  rescaled_x4_Γ3_hist, "Integral": integral_Γ3,
            "All Events": np.transpose(np.array([rescaled_μ2_Γ3_hist,
                                                 rescaled_x3_Γ3_hist,
                                                 rescaled_x4_Γ3_hist])) }


#######################################################
##################     e+e- --> γμ1μ1'    #############
##################  d4σ_deγdcosθγdμ1dμ1p  #############
##################       with VEGAS       #############
#######################################################
def run_MC_dσ_deγdcosθγdμdμp(NPoints, sqrtS, μ0, μϼ):
    print('Running dσ/deγdcosθγdμdμp MC for √s = ' + str(sqrtS) + ' ... ')
    ############################ Functions ########################## 
    d4σ_deγdcosθγdμ1dμ1p_weights = partial(WIC.d4σ_deγdcosθγdμ1dμ1p, sqrtS=sqrtS, μ0=μ0, μϼ=μϼ)
    d4σ_deγdcosθγdμ1dμ1p_v = partial(WIC.d4σ_deγdcosθγdμ1dμ1p_v, sqrtS=sqrtS, μ0=μ0, μϼ=μϼ)
    ########################## Training Vegas ... ################################
    integral = 0.0
#     θc=0.01
#     eγc=5
    integ = vegas.Integrator([[μ0, sqrtS], [μ0, sqrtS], [0, sqrtS], [-1, 1]])
#     NPoints_d3σ = 50000
    result = integ(d4σ_deγdcosθγdμ1dμ1p_v, nitn=100, neval=NPoints, alpha=0.005) ############# train !
    result_error = result.sdev 
    result_integral = result.mean
#     print(result_integral, result_error)
    μ1_σ_hist = []; μ1p_σ_hist = []; eγ_σ_hist = []; cosθγ_σ_hist = [];
    y_histo = []; wgt_array = []; hcube_histo = [];
    for x, y, wgt, hcube in integ.random(yield_hcube=True, yield_y=True):
        wgt_array.append(1/wgt)
        μ1_σ_hist.append(x[0]) 
        μ1p_σ_hist.append(x[1])
        eγ_σ_hist.append(x[2])
        cosθγ_σ_hist.append(x[3])
        y_histo.append(y[0])
        hcube_histo.append(hcube)
        integral += wgt * d4σ_deγdcosθγdμ1dμ1p_v(x)
    print(np.shape(μ1_σ_hist), np.shape(μ1p_σ_hist), np.shape(eγ_σ_hist), np.shape(cosθγ_σ_hist))
    ################# Rescaling Weights & Cutting Events Probabilistically ################
    hcube_counter = []
    for i in range(max(hcube_histo)+1):          ## Loops thru hypercubes
        if i%5000==0: print(str(i) + ' of ' + str(max(hcube_histo)+1))
        count = np.sum(np.array(hcube_histo)==i) ## Define count = number of points in hypercube i
        temp = np.full((count), count)           ## Define temp = [count, count, ... ('count' times)]
        hcube_counter = np.append(hcube_counter, temp)
        hcube_counter.flatten()
    prob = np.array(hcube_counter)/(len(μ1_σ_hist))
    raw_weights = np.array(list(map(d4σ_deγdcosθγdμ1dμ1p_weights, μ1_σ_hist, μ1p_σ_hist, eγ_σ_hist, cosθγ_σ_hist))) / (prob)
    rescaled_raw_weights = raw_weights/max(raw_weights)
    my_random_numbers = np.random.uniform(low=0, high=1, size=len(rescaled_raw_weights))
    rescaled_raw_histo_boolean = rescaled_raw_weights>my_random_numbers
    rescaled_μ1_σ_hist = np.array(μ1_σ_hist)[rescaled_raw_histo_boolean]
    rescaled_μ1p_σ_hist = np.array(μ1p_σ_hist)[rescaled_raw_histo_boolean]
    rescaled_eγ_σ_hist = np.array(eγ_σ_hist)[rescaled_raw_histo_boolean]
    rescaled_cosθγ_σ_hist = np.array(cosθγ_σ_hist)[rescaled_raw_histo_boolean]
    print(np.shape(μ1_σ_hist))
    print(np.shape(rescaled_μ1_σ_hist))
    #######################################################################################
    print('Returning μ1, μ1p Events, eγ Events, cosθγ Events, Integral')
#     print(integral)
    return {"μ1 Events": rescaled_μ1_σ_hist, 
            "μ1p Events": rescaled_μ1p_σ_hist, 
            "eγ Events": rescaled_eγ_σ_hist,
            "cosθγ Events": rescaled_cosθγ_σ_hist,
            "Integral":  integral, 
            "All Events": np.transpose(np.array([rescaled_μ1_σ_hist, 
                                                 rescaled_μ1p_σ_hist,
                                                 rescaled_eγ_σ_hist, 
                                                 rescaled_cosθγ_σ_hist])),
            "Error":  result_error }


#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#
####################################################### SAMPLING & RESCALING ##################################################
#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#


#############################
#### Kinematic functions ####
#############################
def E2(μ1, μ2):
    return (-mZ**2 + μ1**2 + μ2**2)/(2*μ1)
def EZ(μ1, μ2):
    return (mZ**2 + μ1**2 - μ2**2)/(2*μ1)
def pZ(μ1, μ2):
    return np.sqrt((mZ-μ1-μ2)*(mZ+μ1-μ2)*(mZ-μ1+μ2)*(mZ+μ1+μ2))/(2*μ1)
def γ(v):
    return 1/np.sqrt(1-v**2)
def Boost(vx, vy, vz):
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    return np.array([
        [γ(v)    ,  -γ(v)*vx            , -γ(v)*vy             , -γ(v)*vz             ],
        [-γ(v)*vx,  1+(γ(v)-1)*(vx/v)**2, (γ(v)-1)*vx*vy/(v**2), (γ(v)-1)*vx*vz/(v**2)],
        [-γ(v)*vy, (γ(v)-1)*vx*vy/(v**2),  1+(γ(v)-1)*(vy/v)**2, (γ(v)-1)*vy*vz/(v**2)],
        [-γ(v)*vz, (γ(v)-1)*vx*vz/(v**2), (γ(v)-1)*vy*vz/(v**2), 1+(γ(v)-1)*(vz/v)**2],
                    ])
def Rotate(θ, φ):
    return np.array([
        [         1         ,          0         ,     0     ,          0         ],
        [         0         , np.cos(φ)*np.cos(θ), -np.sin(φ), np.cos(φ)*np.sin(θ)],
        [         0         , np.sin(φ)*np.cos(θ),  np.cos(φ), np.sin(φ)*np.sin(θ)],
        [         0         ,    -np.sin(θ)      ,     0     ,      np.cos(θ)     ]
    ])


#############################
#### Rescaling functions ####
#############################
## Rescaling dΓ/dμ2 for 2-body ##
def squeeze_2body(μ2star, μ1star, μ1):
    return (μ2star-100)*(μ1star-100)/(μ1-100) + 100 

# Rescaling d3Γ/dμ2dx3dx4 for 3-body ##
def rescale_μ2(μ1_actual, μ1, μ2, μ0):
    return ((μ1_actual - μ0)/(μ1 - μ0))*(μ2 - μ0) + μ0
def a_scale(μ1_actual, p2, E3, E4, μ2_rescaled):
    return (μ1_actual*(E3 + E4) -\
           np.sqrt(((E3 + E4)**2)*(μ2_rescaled**2) + (μ1_actual - μ2_rescaled)*(μ1_actual + μ2_rescaled)*p2**2))/\
           ((E3+E4-p2)*(E3+E4+p2))
def a_scale_mff(μ1_actual, p3x, p3y, p3z, p4x, p4y, p4z, E3, E4, μ2_rescaled):
    p3dotp4 = p3x*p4x + p3y*p4y + p3z*p4z
    return (-1/(4*μ1_actual*E3))*\
           (μ2_rescaled**2 - μ1_actual**2 - \
            np.sqrt( (μ1_actual**2 - μ2_rescaled**2 + 2*(E3*E4 - p3dotp4))**2 - 16*E3*E4*μ1_actual**2 ) -\
            2*(E3*E4 - p3dotp4) )



##############################
#### SAMPLING & APPENDING ####
##############################

#################################################
########### ALL KINEMATICS FOR DECAYS ###########
#################################################
def FinalState2body_FVs(μ1, μ1_actual, μ2, prime):   ### (2-body isn't really used...) 
        ## Random angles ## 
        φ1 = np.random.uniform(low=0.0, high=2*np.pi)
        θ1 = np.random.uniform(low=0.0, high=  np.pi)
        ## Rescale μ2 ##
        μ2 = squeeze_2body(μ2, μ1_actual, μ1)
        ## 4-vectors in K' ##
        pZ_mu = np.array([
            EZ(μ1_actual, μ2), 
            pZ(μ1_actual, μ2)*np.sin(θ1)*np.cos(φ1),
            pZ(μ1_actual, μ2)*np.sin(θ1)*np.sin(φ1),
            pZ(μ1_actual, μ2)*np.cos(θ1)
        ])
        p2_mu = np.array([
            E2(μ1_actual,  μ2), 
            -pZ(μ1_actual, μ2)*np.sin(θ1)*np.cos(φ1),
            -pZ(μ1_actual, μ2)*np.sin(θ1)*np.sin(φ1),
            -pZ(μ1_actual, μ2)*np.cos(θ1)
        ])
        ### Boost p2, pZ by v1 ### 
        v1x = MC_Events['p1x'][i]/MC_Events['E1'][i]
        v1y = MC_Events['p1y'][i]/MC_Events['E1'][i]
        v1z = MC_Events['p1z'][i]/MC_Events['E1'][i]
        pZ_mu = np.dot(Boost(-v1x, -v1y, -v1z),pZ_mu) ## Don't boost this ...?????
        p2_mu = np.dot(Boost(-v1x, -v1y, -v1z),p2_mu)
        ### Append **BOOSTED** E2, p2 ### 
        if prime==False:
            MC_Events_μ2.append(μ2)
            MC_Events_E2.append(p2_mu[0])
            MC_Events_p2x.append(p2_mu[1]); 
            MC_Events_p2y.append(p2_mu[2]);
            MC_Events_p2z.append(p2_mu[3]);
        if prime==True:
            MC_Events_μ2p.append(μ2)
            MC_Events_E2p.append(p2_mu[0])
            MC_Events_p2xp.append(p2_mu[1]); 
            MC_Events_p2yp.append(p2_mu[2]);
            MC_Events_p2zp.append(p2_mu[3]);
        ########################################## Z -> ff ######################################
        φ2 = np.random.uniform(low=0.0, high=2*np.pi)
        θ2 = np.random.uniform(low=0.0, high=  np.pi)
        ## 4-vectors in K'' ##
        p2a_mu = np.array([
            mZ/2, (mZ/2)*np.sin(θ2)*np.cos(φ2), (mZ/2)*np.sin(θ2)*np.sin(φ2), (mZ/2)*np.cos(θ2)
        ])
        p2b_mu = np.array([
            mZ/2, -(mZ/2)*np.sin(θ2)*np.cos(φ2), -(mZ/2)*np.sin(θ2)*np.sin(φ2), -(mZ/2)*np.cos(θ2)
        ])
        ## Boost p2a, p2b by vZ ##  ## which is already boosted by v1 ... 
        vZx = pZ_mu[1]/pZ_mu[0]; vZy = pZ_mu[2]/pZ_mu[0]; vZz = pZ_mu[3]/pZ_mu[0]
        p2a_mu = np.dot(Boost(-vZx, -vZy, -vZz), p2a_mu)
        p2b_mu = np.dot(Boost(-vZx, -vZy, -vZz), p2b_mu)
        ## Boost p3, p4 by v1 ## 
#         p3_mu = np.dot(Boost(-v1x, -v1y, -v1z), p3_mu)
#         p4_mu = np.dot(Boost(-v1x, -v1y, -v1z), p4_mu)
        ### Append **BOOSTED** E3, p3, E4, p4 ### 
        if prime==False:
            MC_Events_E2a.append(p2a_mu[0]);  MC_Events_E2b.append(p2b_mu[0]);
            MC_Events_p2ax.append(p2a_mu[1]); MC_Events_p2bx.append(p2b_mu[1]);
            MC_Events_p2ay.append(p2a_mu[2]); MC_Events_p2by.append(p2b_mu[2]);
            MC_Events_p2az.append(p2a_mu[3]); MC_Events_p2bz.append(p2b_mu[3]);
        if prime==True:
            MC_Events_E2ap.append(p2a_mu[0]);  MC_Events_E2bp.append(p2b_mu[0]);
            MC_Events_p2axp.append(p2a_mu[1]); MC_Events_p2bxp.append(p2b_mu[1]);
            MC_Events_p2ayp.append(p2a_mu[2]); MC_Events_p2byp.append(p2b_mu[2]);
            MC_Events_p2azp.append(p2a_mu[3]); MC_Events_p2bzp.append(p2b_mu[3]);
def FinalState3body_FVs(μ1, μ1_actual, prime, n, temp_event, threshold, Γ3_interp, μ2x3x4_Γ3_hist):
        ################################## 
        ## Sample a random (μ2, x3, x4) ##
        ####################################
#         μ2x3x4_Γ3_hist = d3Γ3_ds[IndOfM(μ1)]['All Events']
        random_int = random.randint(0,len(μ2x3x4_Γ3_hist))
        μ2 = μ2x3x4_Γ3_hist[random_int][0]
        x3 = μ2x3x4_Γ3_hist[random_int][1]
        x4 = μ2x3x4_Γ3_hist[random_int][2]
#         if μ2 > 103: Γ3_integral = Γ3_interp(μ2)
#         else :       Γ3_integral = 0
#         print('μ2 is: ' + str(μ2))
        ####################################
        ## Kinematics ## 
        ####################################
        ### Random angles ### 
        φ1 = np.random.uniform(low=0.0, high=2*np.pi)
        θ1 = np.random.uniform(low=0.0, high=  np.pi)
        ### Define p2, E3, E4 ###
        p2 = np.sqrt((μ1*(1 - (x3/2) - (x4/2)))**2 - μ2**2)
        E3 = x3*μ1/2
        E4 = x4*μ1/2
        ### Define angles α, x, y ###
        α = np.arccos( (p2**2 - E3**2 - E4**2) / (2*E3*E4) )  
        x = np.arctan( ( np.cos(α) + (x3/x4) ) / (np.sin(α)) )
        y = np.arctan( ( np.cos(α) + (x4/x3) ) / (np.sin(α)) )
        ### Fermion 4-vectors in Rest Frame ###
        p2a_mu = np.array([
            E3,  E3*np.sin(np.pi/2 + x), 0, E3*np.cos(np.pi/2 + x)
        ])
        p2b_mu = np.array([
            E4, -E4*np.sin(np.pi/2 + y), 0, E4*np.cos(np.pi/2 + y)
        ])
#         ### Rotate p2a, p2b by θ1, φ1 ###            ### still don't know if this is necessary or not .....
#         p2a_mu = np.dot(Rotate(θ1, φ1), p2a_mu)      ### is a() a function of rest frame things..? 
#         p2b_mu = np.dot(Rotate(θ1, φ1), p2b_mu)
#         # Boost too .. ?? ## 
#         p2a_mu = np.dot(Boost(-vnx, -vny, -vnz), p2a_mu)
#         p2b_mu = np.dot(Boost(-vnx, -vny, -vnz), p2b_mu)
        ### Rescale μ2 ###
        μ2_rescaled = rescale_μ2(μ1_actual, μ1, μ2, 100)
        #### Rescale p2, E3, E4 by the scale factor 'a' ####
        a = a_scale_mff(μ1_actual, 
                        p2a_mu[1], p2a_mu[2], p2a_mu[3], 
                        p2b_mu[1], p2b_mu[2], p2b_mu[3], E3, E4, μ2_rescaled)
        p2x = -(a*p2a_mu[1] + (1/a)*p2b_mu[1])
        p2y = -(a*p2a_mu[2] + (1/a)*p2b_mu[2])
        p2z = -(a*p2a_mu[3] + (1/a)*p2b_mu[3])
        p2 = np.sqrt(p2x**2 + p2y**2 + p2z**2)
        E3 = a*E3
        E4 = (1/a)*E4
        ########################################
        ### NEW 4-vectors!  p2, p2a, p2b ... ###
        ########################################
        ########
        ## p2 ## 
        ########
        p2_mu = np.array([ np.sqrt(p2**2 + μ2_rescaled**2), p2x, p2y, p2z ])
        ### Rotate (?) p2, p2p by φ1, θ1 ###
        p2_mu = np.dot(Rotate(θ1, φ1), p2_mu)
        ### Boost p2, p2p by vn ### 
        if n == 1: recent_frame = -1
        if n  > 1: recent_frame = -3
        vnx = temp_event[recent_frame][4] / temp_event[recent_frame][3]
        vny = temp_event[recent_frame][5] / temp_event[recent_frame][3]
        vnz = temp_event[recent_frame][6] / temp_event[recent_frame][3]
#         print('for n = ' + str(n) + ', last temp_event is    : ' + str(temp_event[recent_frame]))
        if prime==False:
            p2_mu_BoostVn = np.dot(Boost(-vnx, -vny, -vnz),p2_mu)
        if prime==True:
            p2_mup_BoostVn = np.dot(Boost(-vnx, -vny, -vnz),p2_mu)
        ##############
        ## p2a, p2b ## 
        ##############
#         α = np.arccos( (p2_mu[0]**2 - μ2_rescaled**2 - E3**2 - E4**2) / (2*E3*E4) )  
#         x = np.arctan( ( np.cos(α) + (x3/x4) ) / (np.sin(α)) )
#         y = np.arctan( ( np.cos(α) + (x4/x3) ) / (np.sin(α)) )
        p2a_mu = np.array([
            E3,  E3*np.sin(np.pi/2 + x), 0, E3*np.cos(np.pi/2 + x)
        ])
        p2b_mu = np.array([
            E4, -E4*np.sin(np.pi/2 + y), 0, E4*np.cos(np.pi/2 + y)
        ]) 
        ### Rotate p3, p4 by θ1, φ1 ###
        p2a_mu = np.dot(Rotate(θ1, φ1), p2a_mu)
        p2b_mu = np.dot(Rotate(θ1, φ1), p2b_mu)
        ### Boost p3, p4 by vn ###
        if prime==False:
            p2a_mu_BoostVn = np.dot(Boost(-vnx, -vny, -vnz), p2a_mu)
            p2b_mu_BoostVn = np.dot(Boost(-vnx, -vny, -vnz), p2b_mu)
        if prime==True:
            p2a_mup_BoostVn = np.dot(Boost(-vnx, -vny, -vnz), p2a_mu)
            p2b_mup_BoostVn = np.dot(Boost(-vnx, -vny, -vnz), p2b_mu)
        ###########################################
        ####### Add to MC_Events_LHE Array ########
        ###########################################
        pickanumberanynumber = np.random.uniform(low=0, high=1)
        if 0       < pickanumberanynumber < 0.03632: fermionPID = 11
        if 0.03632 < pickanumberanynumber < 0.07294: fermionPID = 13
        if 0.07294 < pickanumberanynumber < 0.10990: fermionPID = 15
        if 0.10990 < pickanumberanynumber < 0.17656: fermionPID = 12
        if 0.17656 < pickanumberanynumber < 0.24322: fermionPID = 14
        if 0.24322 < pickanumberanynumber < 0.30989: fermionPID = 16
        if 0.30989 < pickanumberanynumber < 0.42589: fermionPID = 2  # u  
        if 0.42589 < pickanumberanynumber < 0.54589: fermionPID = 4  # c
        if 0.54589 < pickanumberanynumber < 0.70189: fermionPID = 1  # d
        if 0.70189 < pickanumberanynumber < 0.85789: fermionPID = 3  # s
        if 0.85789 < pickanumberanynumber < 1      : fermionPID = 5  # b
        if μ2_rescaled > threshold: 
            decay = 1
            τ = hbar/Γ3_interp(μ2_rescaled)
        if μ2_rescaled < threshold: 
            decay = 0
            τ = np.inf
        if prime==False:
            temp_event.append(['PID' + str(n+1),       1000000, μ2_rescaled, 
                               p2_mu_BoostVn[0], p2_mu_BoostVn[1], 
                               p2_mu_BoostVn[2], p2_mu_BoostVn[3], decay, τ])
            temp_event.append(['PID' + str(n+1) + 'a',  fermionPID, 0, 
                               p2a_mu_BoostVn[0], p2a_mu_BoostVn[1], 
                               p2a_mu_BoostVn[2], p2a_mu_BoostVn[3], 0, np.inf])
            temp_event.append(['PID' + str(n+1) + 'b', -fermionPID, 0, 
                               p2b_mu_BoostVn[0], p2b_mu_BoostVn[1],
                               p2b_mu_BoostVn[2], p2b_mu_BoostVn[3], 0, np.inf])
        if prime==True:
            temp_event.append(['PID' + str(n+1) + 'p', 1000000, μ2_rescaled, 
                               p2_mup_BoostVn[0], p2_mup_BoostVn[1], 
                               p2_mup_BoostVn[2], p2_mup_BoostVn[3], decay, τ])
            temp_event.append(['PID' + str(n+1) + 'ap',  fermionPID, 0, 
                               p2a_mup_BoostVn[0], p2a_mup_BoostVn[1], 
                               p2a_mup_BoostVn[2], p2a_mup_BoostVn[3], 0, np.inf])
            temp_event.append(['PID' + str(n+1) + 'bp', -fermionPID, 0, 
                               p2b_mup_BoostVn[0], p2b_mup_BoostVn[1],
                               p2b_mup_BoostVn[2], p2b_mup_BoostVn[3], 0, np.inf])


def ProductionFVs(μ1_sample, μ1p_sample, cosθ_sample, temp_event, prime, threshold, Γ3_int, s):
    ## Random Angles
    φ0 = np.random.uniform(low=0.0, high=2*np.pi)
    θ0 = cosθ_sample
    ## Energy
    E1 = (s + μ1_sample**2 - μ1p_sample**2)/(2*np.sqrt(s))
    E1p = (s + μ1p_sample**2 - μ1_sample**2)/(2*np.sqrt(s))
    if prime==False:
        if μ1_sample > threshold: 
            decay = 1
            print('μ1_sample is ' + str(μ1_sample))
            τ = hbar/Γ3_int(μ1_sample)
        if μ1_sample < threshold: 
            decay = 0
            τ = np.inf
        temp_event.append(['PID1', 1000000, μ1_sample, E1,
                        np.sqrt(E1**2 - μ1_sample**2)*np.sin(θ0)*np.cos(φ0),
                        np.sqrt(E1**2 - μ1_sample**2)*np.sin(θ0)*np.sin(φ0),
                        np.sqrt(E1**2 - μ1_sample**2)*np.cos(θ0), decay, τ])
    if prime==True:
        if μ1p_sample > threshold: 
            decay = 1
            τ = hbar/Γ3_int(μ1p_sample)
        if μ1p_sample < threshold: 
            decay = 0
            τ = np.inf
        temp_event.append(['PID1p', 1000000, μ1p_sample, E1p,
                        np.sqrt(E1p**2 - μ1p_sample**2)*np.sin(θ0)*np.cos(φ0),
                        np.sqrt(E1p**2 - μ1p_sample**2)*np.sin(θ0)*np.sin(φ0),
                        np.sqrt(E1p**2 - μ1p_sample**2)*np.cos(θ0), decay, τ])
        
        