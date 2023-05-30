import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import quad, dblquad, tplquad, simpson


#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#
####################################################### PHYSICS FORMULAE ######################################################
#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#******#

###############################################################################################################################
#################################################### Physical Constants #######################################################
###############################################################################################################################

mZ = 91.2; ZRate = 2.5; cosθ = 0.87; sinθ = 0.48; g = 0.65; gp = 0.36; ## At top mass  
ν_charges = (1/2)**2
l_charges = sinθ**4 + (sinθ**2 - (1/2))**2
u_charges = ((-2/3)*sinθ**2)**2 + ((1/2) - (2/3)*sinθ**2)**2  
d_charges = ((1/3)*sinθ**2)**2 + (-(1/2) + (1/3)*sinθ**2)**2
cf = ((1/2)*(g/cosθ)**2)*(3*ν_charges + 3*l_charges + 3*2*u_charges + 3*3*d_charges);   ## 1/2 in front is bc 
ce = ((1/2)*(g/cosθ)**2)*l_charges;                                                     ##  2(gV^2 + gA^2) = cL^2 + cR^2
ceγ = np.sqrt(4*np.pi / 137)
gZ = np.sqrt(g**2 + gp**2); 
SinSquaredAlpha = 0.1;  GeVm2toAttobarn = 0.3894*(10**15);

##############################################################################################################################
###################################################### Cross Section #########################################################
##############################################################################################################################

#################
## d2σ_dμ1dμ1p ## 
#################
σCOEFF = ( (GeVm2toAttobarn*ce*(SinSquaredAlpha*gZ)**2)/(8*np.pi*(np.pi)**2) ) * (1/4); ## 1/4 to average over spins 

def σμ(sqrtS, μ1, μ2):
    return (( (sqrtS**4 - 2*sqrtS**2*(μ1**2 + μ2**2) + (μ1**2 - μ2**2)**2 ) / (4*sqrtS**2)  )**(3/2)) /\
           (sqrtS*(sqrtS**2 - mZ**2)**2)

def d2σ_dμ1dμ1p(μ1, μ1p, sqrtS, μ0, μϼ):
    if μ0 < μ1 < sqrtS - μ0 and μ0 < μ1p < sqrtS - μ1: 
        return σCOEFF*μϼ(μ1)*μϼ(μ1p)*σμ(sqrtS, μ1, μ1p)
    else:
        return 0
def d2σ_dμ1dμ1p_v(X, sqrtS, μ0, μϼ):
    μ1   = X[0]
    μ1p  = X[1]
    return d2σ_dμ1dμ1p(μ1, μ1p, sqrtS, μ0, μϼ) 

################
##  dσ_dcosθ  ## 
################
def dσ_dcosθ(x):
    return 1-x**2
def dσ_dcosθ_v(X):
    x   = X[0]
    return dσ_dcosθ(x) 

    
###########
## dσ_dμ ## 
###########
def integrand_dσdμ(μ2, μ1, sqrtS, μϼ):
        return μϼ(μ2)*σμ(sqrtS, μ1, μ2)
def dσdμ(μ1, μ0, sqrtS, μϼ):
    return σCOEFF*μϼ(μ1)*quad(integrand_dσdμ, μ0, sqrtS-μ1, args=(μ1,sqrtS,μϼ))[0]


#######
## σ ## 
#######
def σ(sqrtS, μ0, μϼ):
    return quad(dσdμ, μ0, sqrtS-μ0, args=(μ0,sqrtS,μϼ))[0]


##########################
## d4σ_deγdcosθγdμ1dμ1p ##   ( KK production w/ Initial state radiation ) 
##########################

σISR_COEFF = GeVm2toAttobarn*ce*((ceγ*SinSquaredAlpha*gZ)/(2))**2 * (1/4); ## 1/4 to average over spins 

def Ratio(μ1,sqrtS):
    return (2*μ1/sqrtS)**2
def R2(μ1p,sqrtS):
    return (2*μ1p/sqrtS)**2
def dR(μ1, μ1p,sqrtS):
    return (2*μ1/sqrtS)**2 - (2*μ1p/sqrtS)**2
def σISRμ(μ1, μ1p, eγ, cosθγ, sqrtS):
    R1_=Ratio(μ1,sqrtS)
    R2_=Ratio(μ1p,sqrtS)
    ΔR_=R1_-R2_
    ΣR_=R1_+R2_
    xγ = eγ/(sqrtS/2)
    θγ = np.arccos(cosθγ)
    num1  = 2*np.pi**2*(sqrtS**2 / 4)*(1/(np.sin(θγ)**2))*(8*(xγ-1)*ΣR_ + ΔR_**2 + 16*(xγ-1)**2)**(3/2)
    num2  = ((xγ-1)*(xγ+3)*np.cos(2*θγ) + xγ*(3*xγ-10) + 11)
    denom = (3*(xγ-1)**2*xγ*(sqrtS**2*(xγ-1)+mZ**2)**2)
    return num1*num2/denom

             
                                          
def d4σ_deγdcosθγdμ1dμ1p(μ1, μ1p, eγ, cosθγ, sqrtS, μ0, μϼ):
    R1_=Ratio(μ1,sqrtS)
    R2_=Ratio(μ1p,sqrtS)
    ΔR_=R1_-R2_
    ΣR_=R1_+R2_
    e = (sqrtS/2)
    xγ  = eγ/e
    θc=0.1
    xγc = 0.05
    if μ1p < sqrtS-μ1-e*xγc and\
       μ1  < sqrtS-μ0-e*xγc and\
       xγ  > xγc and\
       xγ < 1-(ΣR_/4) - (np.sqrt(R1_*R2_)/2) and\
       -1+θc < cosθγ < 1-θc:
        return σISR_COEFF*(μϼ(μ1)*μϼ(μ1p)/(np.pi**2))*σISRμ(μ1, μ1p, eγ, cosθγ, sqrtS)/(64*(2*np.pi)**5)
    else: 
        return 0
def d4σ_deγdcosθγdμ1dμ1p_v(X, sqrtS, μ0, μϼ):
    μ1    = X[0]
    μ1p   = X[1]
    eγ    = X[2]
    cosθγ = X[3]
    return d4σ_deγdcosθγdμ1dμ1p(μ1, μ1p, eγ, cosθγ, sqrtS, μ0, μϼ)



###########################################################################################################################
########################################################### Rate ##########################################################
###########################################################################################################################
def R(μ1,μ2):
    return (μ2/μ1)**2
def R0(μ1,μ0):
    return (μ0/μ1)**2
def MSquared3body(x3, x4, μ1, μ2): 
    return (2*(μ1**4)*(cf*(SinSquaredAlpha*gZ)**2)*(1 - R(μ1, μ2) + x3*x4 - x3 - x4)) /\
           (np.abs((μ1**2)*(x3 + x4 + R(μ1, μ2) - 1) - mZ**2 + mZ*ZRate*1j)**2) 
def MSquared2body(μ1, μ2): 
    return (((SinSquaredAlpha*gZ)/2)**2)*(((μ1**2 - μ2**2)/mZ)**2 - 2*(μ1**2 + μ2**2) + mZ**2)

###########
## dΓ_dμ ## 
###########

######### 3 body #########
def d3Γ3_dμ2dx3dx4(μ2, x3, x4, μ1, μ0, μϼ, Δm2):
    if μ0 <= μ2 < μ1 and\
       1 - (x3/2) - (x4/2) >= np.sqrt(R(μ1,μ2)) and\
       x3 + x4         >= 1 - R(μ1,μ2) and\
       x3 + x4 - x3*x4 <= 1 - R(μ1,μ2):
        return ((μϼ(μ1)/μ1) * Δm2)*(μ1/(256*np.pi**4))*μϼ(μ2)*MSquared3body(x3, x4, μ1, μ2)
    else:
        return 0
def d3Γ3_dμ2dx3dx4_v(X, μ1, μ0, μϼ, Δm2):
    μ2 = X[0]
    x3 = X[1]
    x4 = X[2]
    return d3Γ3_dμ2dx3dx4(μ2, x3, x4, μ1, μ0, μϼ, Δm2)


######### 2 body #########
def dΓ2_dμ2(μ2, μ1, μ0, μϼ):
    if μ0 < μ2 < μ1-mZ: 
        return ((μϼ(μ2))/(2*μ1*4*np.pi*μ1*np.pi))*np.sqrt(((μ1-μ2-mZ)*(μ1+μ2-mZ)*(μ1-μ2+mZ)*(μ1+μ2+mZ))/\
               (4*μ1**2))*MSquared2body(μ1, μ2)
    else:
        return 0
def dΓ2_dμ2_v(X, μ1, μ0, μϼ):
    μ2  = X[0]
    return dΓ2_dμ2(μ2, μ1, μ0, μϼ)
    
######### Piecewise #########
def dΓdμ(μ2, μ1, μϼ): 
    if μ2 < μ1 - mZ - 3: return dΓ2_dμ2(μ2, μ1, μϼ)
    if μ2 >= μ1 - mZ- 3: return dΓdμ_3body(μ2, μ1, μϼ)   
    

############
## dΓ_dE3 ## 
############
def dΓdE3_3body_integrand(μ2, x4, x3, μ1, μ0, μϼ):        
    return (μ1/2)*((μϼ(μ2))/(2*μ1*32*np.pi**4))*MSquared3body(x3, x4, μ1, μ2)
def dΓdE3_3body(E3, μ1, μ0, μϼ): 
    x3=2*E3/μ1
    return dblquad(dΓdE3_3body_integrand, (1-x3-R0(μ1,μ0))/(1-x3), (1-x3-R0(μ1,μ0)), 
                                          lambda x4: μ1*np.sqrt(1-x3-x4), lambda x4: μ1*np.sqrt(1-x3-x4+x3*x4), 
                                          args=[x3, μ1, μ0, μϼ])[0]

#######
## Γ ## 
#######
def Γ_3body(μ1, μ0, μϼ):
    return quad(dΓdμ_3body, μ0, μ1, args=(μ1,μϼ))[0]
def Γ_2body(μ1, μ0, μϼ):
    return quad(dΓdμ_2body, μ0, μ1-mZ, args=(μ1,μϼ))[0]
def Γ(μ1, μ0, μϼ):
    if μ1 < μ0+mZ+14:
        return quad(dΓdμ_3body, μ0, μ1, args=(μ1,μϼ))[0]
    if μ1 >= μ0+mZ+14:
        return quad(dΓdμ_2body, μ0, μ1-mZ, args=(μ1,μϼ))[0]

