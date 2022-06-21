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

mZ = 91.2; cosθ = 0.87; sinθ = 0.48; g = 0.65; gp = 0.36; ## At top mass  
ν_charges = 1/2
l_charges = np.sqrt(sinθ**4 + (sinθ**2 - (1/2))**2)
u_charges = np.sqrt( ((-2/3)*sinθ**2)**2 + ((1/2) - (2/3)*sinθ**2)**2 ) 
d_charges = np.sqrt( ((1/3)*sinθ**2)**2 + (-(1/2) + (1/3)*sinθ**2)**2 )
cf = ((g/cosθ)**2)*(3*ν_charges**2 + 3*l_charges**2 + 3*2*u_charges**2 + 3*3*d_charges**2)/2; 
ce = ((g/cosθ)**2)*(l_charges**2)/2;
gZ = np.sqrt(g**2 + gp**2); 
SinSquaredAlpha = 0.1;  GeVm2toAttobarn = 0.3894*(10**15);

##############################################################################################################################
###################################################### Cross Section #########################################################
##############################################################################################################################

#################
## d2σ_dμ1dμ1p ## 
#################
σCOEFF = ( (GeVm2toAttobarn*ce*(SinSquaredAlpha*gZ)**2)/(6*np.pi*(np.pi)**2) )*(1/4)*(1/2);
def σμ(sqrtS, μ1, μ2):
    return (( ((sqrtS**2 + μ1**2 - μ2**2)/(2*sqrtS))**2 - μ1**2 )**(3/2)) / (sqrtS*(sqrtS**2 - mZ**2)**2)
def d2σ_dμ1dμ1p(μ1, μ1p, sqrtS, μ0, μϼ):
    if μ0 < μ1 < sqrtS - μ0 and μ0 < μ1p < sqrtS - μ1: 
        return σCOEFF*μϼ(μ1)*μϼ(μ1p)*σμ(sqrtS, μ1, μ1p)
    else:
        return 0
def d2σ_dμ1dμ1p_v(X, sqrtS, μ0, μϼ):
    μ1   = X[0]
    μ1p  = X[1]
    return d2σ_dμ1dμ1p(μ1, μ1p, sqrtS, μ0, μϼ) 
    
###########
## dσ_dμ ## 
###########
def integrand_dσdμ(μ2, μ1, sqrtS, μϼ):
        return μϼ(μ2)*σμ(sqrtS, μ1, μ2)
def dσdμ(μ1, μ0, sqrtS, μϼ):
    return σCOEFF*μϼ(μ1)*quad(integrand_dσdμ, μ0, sqrtS-μ1, args=(μ1,sqrtS,μϼ))[0]


################
## d2σ_dμ1dv1 ##  
################
def MaxV_2body(μ0, μ1):
    return np.sqrt((μ0-μ1-mZ)*(μ0-μ1+mZ)*(μ0+μ1-mZ)*(μ0+μ1+mZ))/(μ0**2+μ1**2-mZ**2)
def MaxV_3body(μ1, μ0):
    return (μ1**2 - μ0**2)/(μ1**2 + μ0**2)

def μ1p(μ1, v1, sqrtS):
    return np.sqrt(μ1**2 + sqrtS**2 - 2*μ1*sqrtS*γ(v1))
def dμ1pdv1(μ1, v1, sqrtS):
    return -(μ1*sqrtS*v1*γ(v1)**3)/μ1p(μ1, v1, sqrtS)
def d2σdμ1dv1(μ1, v1, sqrtS, μϼ):
        return σCOEFF*μϼ(μ1)*μϼ(μ1p(μ1, v1, sqrtS))*σμ(sqrtS, μ1, μ1p(μ1, v1, sqrtS))*dμ1pdv1(μ1, v1, sqrtS)

###########
## dσ_dv ## 
###########
def μ1(v1, μ2, sqrtS):
    return γ(v1)*sqrtS - np.sqrt(μ2**2 + (sqrtS**2)*(γ(v1)**2 - 1) )
def dμdv(v1, μ2, sqrtS): 
    return sqrtS*v1*(γ(v1)**3) - (((sqrtS**2)*v1*(γ(v1)**4))/(np.sqrt(μ2**2 + (sqrtS**2)*(γ(v1)**2 - 1))))
def integrand_dσdv(μ2, v1, sqrtS, μϼ):
    return -μϼ(μ1(v1, μ2, sqrtS))*μϼ(μ2)*σμ(sqrtS, μ1(v1, μ2, sqrtS), μ2)*dμdv(v1, μ2, sqrtS)
def dσdv(v1, sqrtS, μ0, μϼ):
    return σCOEFF*quad(integrand_dσdv, μ0, np.sqrt(μ0**2 - 2*μ0*γ(v1)*sqrtS + sqrtS**2) , args=(v1,sqrtS,μϼ))[0]

#######
## σ ## 
#######
def σ(sqrtS, μ0, μϼ):
    return quad(dσdμ, μ0, sqrtS-μ0, args=(μ0,sqrtS,μϼ))[0]


###########################################################################################################################
########################################################### Rate ##########################################################
###########################################################################################################################
def R(μ1,μ2):
    return (μ2/μ1)**2
def R0(μ1,μ0):
    return (μ0/μ1)**2
def MSquared3body(x3, x4, μ1, μ2): 
    return (2*(μ1**4)*(cf*(SinSquaredAlpha*gZ)**2)*(1 - R(μ1, μ2) + x3*x4 - x3 - x4)) /\
           (np.abs((μ1**2)*(x3 + x4 + R(μ1, μ2) - 1) - mZ**2 + mZ*2.5*1j)**2) 
def MSquared2body(μ1, μ2): 
    return (((SinSquaredAlpha*gZ)/2)**2)*(((μ1**2 - μ2**2)/mZ)**2 - 2*(μ1**2 + μ2**2) + mZ**2)

###########
## dΓ_dμ ## 
###########

######### 3 body #########
def d3Γ3_dμ2dx3dx4(μ2, x3, x4, μ1, μ0, μϼ):
    if μ0 <= μ2 < μ1 and\
       1 - (x3/2) - (x4/2) >= np.sqrt(R(μ1,μ2)) and\
       x3 + x4         >= 1 - R(μ1,μ2) and\
       x3 + x4 - x3*x4 <= 1 - R(μ1,μ2):
        return (μ1/(256*np.pi**4))*μϼ(μ2)*MSquared3body(x3, x4, μ1, μ2)
    else:
        return 0
def d3Γ3_dμ2dx3dx4_v(X, μ1, μ0, μϼ):
    μ2 = X[0]
    x3 = X[1]
    x4 = X[2]
    return d3Γ3_dμ2dx3dx4(μ2, x3, x4, μ1, μ0, μϼ)


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
    
###########
## dΓ_dv ## 
###########
def μ2(μ1, v2):
    return μ1*γ(v2) - np.sqrt(mZ**2 + (μ1*v2*γ(v2))**2)
def dμ2dv2_2body(μ1, v2):
    return -(μ1*v2*(γ(v2)**3) - (((μ1**2)*v2*(γ(v2)**4))/np.sqrt(mZ**2 + (μ1*v2*γ(v2))**2)))
def dΓdv2_2body(v2, μ1, μϼ):
    return dμ2dv2_2body(μ1, v2)*dΓdμ_2body(μ2(μ1, v2), μ1, μϼ)

def μ2_3body(μ1, v2, x3, x4):
    return μ1*(1-(x3/2)-(x4/2))/γ(v2)
def dμ2dv2_3body(μ1, v2, x3, x4):
    return -μ1*((x3/2)+(x4/2)-1)*γ(v2)*v2
def dΓdv2_3body_integrand(x3, x4, v2, μ1, μϼ):
    return (1/2)*((μ1**2*μϼ(μ2_3body(μ1, v2, x3, x4)))/(2*μ1*np.pi*128*np.pi**3))*dμ2dv2_3body(μ1, v2, x3, x4)*MSquared3body(x3, x4, μ1, μ2_3body(μ1, v2, x3, x4))
def lower_lim(v2, μ1, μ0):
    return 2-(2*γ(v2)*μ0/μ1)
def upper_lim(v2):
    return 2*v2/(v2+1)
def dΓdv2_3body(v2, μ1, μ0, μϼ):
    return dblquad(dΓdv2_3body_integrand, 0, 2*γ(v2)*v2*μ0/μ1,
                                          lambda x3: lower_lim(v2, μ1, μ0)-x3, lambda x3: (2*γ(v2)*v2*μ0/μ1)-x3,
                                          args=[v2, μ1, μϼ] )[0]

#### Using Step Function ? #### 
def MaxV_2body(μ0, μ1):
    return np.sqrt((μ0-μ1-mZ)*(μ0-μ1+mZ)*(μ0+μ1-mZ)*(μ0+μ1+mZ))/(μ0**2+μ1**2-mZ**2)
def MaxV_3body(μ1, μ0):
    return (μ1**2 - μ0**2)/(μ1**2 + μ0**2)
def dΓdv2_3body_step_integrand(x3, x4,  μ1, v2, μ0, μϼ):
    if      x3 + x4 >= 1 - R(μ1, μ2_3body(μ1, v2, x3, x4)) \
        and x3 + x4 - x3*x4 <= 1 - R(μ1, μ2_3body(μ1, v2, x3, x4)) \
        and μ1 >= μ0 \
        and μ2_3body(μ1, v2, x3, x4) >= μ0:
        return (1/2)*((μ1**2*μϼ(μ2_3body(μ1, v2, x3, x4)))/(2*μ1*np.pi*128*np.pi**3))*\
               dμ2dv2_3body(μ1, v2, x3, x4)*MSquared3body(x3, x4, μ1, μ2_3body(μ1, v2, x3, x4))
    else:
        return 0
def dΓdv2_3body_step(v2, μ1, μ0, μϼ):
    return dblquad(dΓdv2_3body_step_integrand, 0, 2, 0, 2, args=[μ1, v2, μ0, μϼ])[0]

def dΓdv2(v2, μ1, μ0, μϼ):
    if v2 < MaxV_2body(μ0, μ1) or μ0 + mZ <= μ1: 
        return dΓdv2_2body(v2, μ1, μϼ)
    if v2 >= MaxV_2body(μ0, μ1) or μ0 + mZ > μ1: 
        return dΓdv2_3body(v2, μ1, μ0, μϼ)
def dΓdv2_step(v2, μ1, μ0, μϼ):
    if v2 < MaxV_2body(μ0, μ1) or μ0 + mZ <= μ1: 
        return dΓdv2_2body(v2, μ1, μϼ)
    if v2 >= MaxV_2body(μ0, μ1) or μ0 + mZ > μ1: 
        return dΓdv2_3body_step(v2, μ1, μ0, μϼ)


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


###########################################################################################################################
######################################################### Cascade #########################################################
###########################################################################################################################

############
## dσdμ_n ## 
############

# With Vegas
def dσdμ_n_ee_INTEGRAND_vegas(X):
    μ1 = X[0]
    return dσdμ_nm1_int(μ1)*float(dΓdμ_int(μ2, μ1))/Γ_int(μ1)

# Without Vegas
def dσdμ_n_ee_INTEGRAND(μ1, μ2, dσdμ_nm1_int, dΓdμ_int, Γ_int):
    return dσdμ_nm1_int(μ1)*dΓdμ_int(μ2, μ1)/Γ_int(μ1)
def dσdμ_n(μ2, μ0, sqrtS, dσdμ_nm1_int, dΓdμ_int, Γ_int):
    return quad(dσdμ_n_ee_INTEGRAND, μ2, sqrtS - μ0, args=(μ2, dσdμ_nm1_int, dΓdμ_int, Γ_int))[0]


############
## dσdv_n ## 
############

## Boost
def v2kp_sh(v1, v2k, Θ): 
    return  -( (2*v1*(v2k**2-1)*np.cos(Θ) + 
            np.sqrt(2)*np.sqrt((v1**2 - 1)*(v1**2 + (v1**2 - 2)*v2k**2 + v1**2*(v2k**2-1)*np.cos(2*Θ))))/
            (-2+2*v1**2*(v2k**2*np.cos(Θ)**2 + np.sin(Θ)**2)) ) 
def v2kp(v1, v2k, Θ):
    return np.sqrt(v1**2 + v2k**2 - 2*v1*v2k*np.cos(Θ) - (v1*v2k*np.sin(Θ))**2)/(1 - v1*v2k*np.cos(Θ))
def dv2kp_dv2k(v1, v2k, Θ):
    return ((-1+v1**2)*(-v2k+v1*np.cos(Θ)))/\
           ((-1+v1*v2k*np.cos(Θ))**2 * np.sqrt(v1**2+v2k**2-v1*v2k*(2*np.cos(Θ)+v1*v2k*np.sin(Θ)**2)))
#     return ( (1+v1*v2kp*np.cos(Θ))**2 * np.sqrt(v1**2 + v2kp**2 + v1*v2kp*(2*np.cos(Θ)-v1*v2kp*np.sin(Θ)**2)) )/\
#            ( (1-v1**2)*(v2kp+v1*np.cos(Θ)) )



### Boost ###
def γ(v):
    return 1/np.sqrt(1-v**2)



