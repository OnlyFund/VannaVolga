from SABR import BaseNormalSABR
import numpy as np

class Hagan2002NormalSABR(BaseNormalSABR):

    def alpha(self):

        f,s,t,v_atm_n = self.f, self.shift, self.t , self.v_atm_n
        beta,rho,volvol = self.beta,self.rho,self.volvol
        return alpha_normal(f+s, t, v_atm_n, beta, rho, volvol)

    def normal_vol(self,k):

        f,s,t = self.f ,self.shift,self.t
        beta,rho,volvol = self.beta,self.rho,self.volvol
        alpha=self.alpha()
        v_n = normal_vol(k+s,f+s,t,alpha,beta,rho,volvol)
        return v_n

def normal_vol(k,f,t,alpha,beta,rho,volvol):

    f_av = np.sqrt(f*k)
    a = -beta*(2-beta)*(alpha**2)/(24*f_av**(2-2*beta))
    b = rho*alpha*volvol*beta/(4*f_av**(1-beta))
    c = (2-3*rho**2)*volvol**2/24


    def _x(rho,z):
        a = (1-2*rho*z+z**2)**0.5+z-rho
        b = 1-rho
        return np.log(a/b)

    def _f_minus_k_ratio(f,k,beta):
        eps = 1e-07
        if abs(f-k)>eps:
            if abs(1-beta)>eps:
                return (1-beta)*(f-k)/(f**(1-beta)-k**(1-beta))
            else:
                return (f-k)/np.log(f/k)
        else:
            return k**beta

    def zeta_over_x_of_zeta(k,f,t,alpha,beta,rho,volvol):
        eps = 1e-07
        f_av = np.sqrt(f*k)
        zeta = volvol*(f-k)/(alpha*f_av**beta)
        if abs(zeta)>eps:
            return zeta/_x(rho,zeta)
        else:
            return 1
    FMKR = _f_minus_k_ratio(f,k,beta)
    ZXZ = zeta_over_x_of_zeta(k,f,t,alpha,beta,rho,volvol)
    v_n = alpha*FMKR*ZXZ*(1+(a+b+c)*t)
    return v_n

def alpha_normal(f,t,v_atm_n,beta,rho,volvol):
    f_ = f**(1-beta)
    p = [-beta*(2-beta)/(24*f_**2)*t*f**beta,
         t*f**beta*rho*beta*volvol/(4*f_),
         (1+t*volvol**2*(2-3*rho**2)/24)*f**beta,
         -v_atm_n]
    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots), np.real(roots))
    # Note: the double real roots case is not tested
    alpha_first_guess = v_atm_n * f ** (-beta)
    i_min = np.argmin(np.abs(roots_real - alpha_first_guess))
    return roots_real[i_min]



