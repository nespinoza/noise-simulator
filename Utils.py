import sys
import numpy as np
# Used by FlickerNoiseGenerator
from scipy import randn,fft,ifft,real
from numpy import arange,pi,cos,sin,zeros,double
# Used by ARgenerator
from numpy.random import normal
from numpy import append
# Used by getConf
from numpy import sort
# Used by getMAD
from numpy import abs

# Idea taken from the C code by Paul Burke (paulbourke.net/fractals/noise)
# In the notation of Carter & Winn, beta = gamma.

def FlickerGenerator(n,beta):
    re=zeros(n)
    im=zeros(n)
    for i in range(1,(n/2)+1,1):
        mag = (double(i)+1.0)**(-beta/2.)*randn(1)
        pha = 2.*pi*randn(1)
        re[i] = mag*cos(pha)
        im[i] = mag*sin(pha)
        re[n-i] = re[i]
        im[n-i] = -im[i]
    im[n/2]=0.0
    res=ifft(re+complex(0,1)*im)
    return real(res)

#
# a is an array (of the coefficients)!
#

def ARgenerator(a,sigma,n,burnin=0):
    if(burnin==0):
      burnin=10*len(a) # Burn-in elements!
    w=normal(0,sigma,n+burnin)
    AR=array([])
    s=0.0
    for i in range(n+burnin):
        if(i<len(a)):
          AR=append(AR,w[i])
        else:
          s=0.0
          for j in range(len(a)):
              s=s+a[j]*AR[i-j-1]
          AR=append(AR,s+w[i])
    print 'Measured standard deviation: '+str(sqrt(var(w[burnin:])))
    return AR[burnin:]

def ARMAgenerator(phi,theta,sigma,n,burnin=0):
    l=max(len(phi),len(theta))
    if(burnin==0):
      burnin=10*l # Burn-in elements!
    w=normal(0,sigma,n+burnin)
    ARMA=np.array([])
    s=0.0
    l=max(len(phi),len(theta))
    for i in range(n+burnin):
        if(i<l):
          ARMA=np.append(ARMA,w[i])
        else:
          s=0.0
          for j in range(len(phi)):
              s=s+phi[j]*ARMA[i-j-1]
          for j in range(len(theta)):
              s=s+theta[j]*w[i-j-1]
          ARMA=np.append(ARMA,s+w[i])
    return ARMA[burnin:]

from astroML.time_series import lomb_scargle
def get_LS(t,y,yerr=None):
	if yerr == None:
		yerr = np.ones(len(y))

	min_t = 2.0*np.mean(np.abs(np.diff(t))) # Nyquist freq
	max_t = np.max(t) - np.min(t)

	max_f = 1./min_t
	min_f = 1./max_t

	freqs = np.linspace(min_f,max_f,len(t))

	return freqs,lomb_scargle(t,y,yerr,2.*np.pi*freqs,generalized=True) 

def Fit_ARMA(x,phi,theta,sigma):
    l=max(len(phi),len(theta))
    Fit = array([])
    for i in range(len(x)):
        if(i<l):
          Fit=append(Fit,x[i])
        else:
          s=0.0
          for j in range(len(phi)):
              s=s+phi[j]*Fit[i-j-1]
          Fit=append(Fit,s)
    return Fit
