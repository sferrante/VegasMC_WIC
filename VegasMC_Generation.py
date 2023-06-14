import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import vegas
from functools import partial 
import WIC_Pheno_Formulae as WIC
from WIC_Pheno_Formulae import *

hbar = 6.58*10**(-16)*10**(-9)   ## GeV*s

#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#
################################################### VEGAS INTEGRATION GENERAL  ################################################
#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#
def run_MC(NPoints, parameters, integrand, bounds):
    print('Running ' + str(integrand.__name__) + ' ... ' )
    df_dx = partial(integrand, parameters=parameters)
    integral = 0.0
    integ = vegas.Integrator(bounds)
    result = integ(df_dx, nitn=10, neval=NPoints, alpha=0.0005) ############# train !
    result_error = result.sdev 
    result_integral = result.mean
    print('Integral, Error: ' + str(result_integral) + ' , ' + str(result_error))
    x_hist = []; y_histo = []; wgt_array = []; hcube_histo = [];
    for x, y, wgt, hcube in integ.random(yield_hcube=True, yield_y=True):
        wgt_array.append(1/wgt)
        x_hist.append(np.array(x))
        y_histo.append(y[0])
        hcube_histo.append(hcube)  ## hcube is the *index/label* of the hypercube that the point is in
        integral += wgt * df_dx(x)
    ################# Rescaling Weights & Cutting Events Probabilistically ################
    hcube_counter = []
    for i in range(max(hcube_histo)+1):          ## Loops thru hypercubes
        if i%5000==0: print(str(i) + ' of ' + str(max(hcube_histo)+1))
        count = np.sum(np.array(hcube_histo)==i) ## Define count = number of points in hypercube i
        temp = np.full((count), count)           ## Define temp = [count, count, ... ('count' times)]
        hcube_counter = np.append(hcube_counter, temp)
        hcube_counter.flatten()
    prob = np.array(hcube_counter)/(len(np.transpose(x_hist)[0]))
    raw_weights = np.array(list(map(df_dx, x_hist ))) / (prob)
    rescaled_raw_weights = raw_weights/max(raw_weights)
    my_random_numbers = np.random.uniform(low=0, high=1, size=len(rescaled_raw_weights))
    rescaled_raw_histo_boolean = rescaled_raw_weights>my_random_numbers
    rescaled_x_hist = np.array(x_hist)[rescaled_raw_histo_boolean]
    print('shape of raw hist = ' + str(np.shape(x_hist)))
    print('shape of rescaled hist = ' + str(np.shape(rescaled_x_hist)))
    print('Unweighting Efficiency = ' + str(np.shape(rescaled_x_hist)[0]/np.shape(np.transpose(x_hist)[0])[0]))
    #######################################################################################
    return {"Raw x Events": x_hist, 
            "x Events": rescaled_x_hist,
            "Integral":  integral, 
            "Error":  result_error }


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

#############################################
########### KINEMATICS FOR DECAYS ###########
#############################################

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
#         ### Rotate p2a, p2b by θ1, φ1 ###            
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
#             print('μ1_sample is ' + str(μ1_sample))
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
        
        