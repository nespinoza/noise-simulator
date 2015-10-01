import numpy as np
import Utils

################################# OPTIONS #################################

# Number of datapoints to simulate:
ndata = 100

# Number of time-series to simulate to get the "real", "underlying" PSD:
nsims = 1000

# Noise level of added white noise to the time-series (can be 0 if you want):
sigma = 0.0

# AR part of the ARMA process (leave at 0.0 if you want a MA series):
AR_coefficients = np.array([1.0,-0.9])

# MA part of the ARMA process (leave at 0.0 if you want an AR series):
MA_coefficients = np.array([0.0])

# Standard-deviation of the white noise driving the ARMA process:
sigma_ARMA = 0.5

###########################################################################

AR_order = len(np.where(np.abs(AR_coefficients)>0.0)[0])
MA_order = len(np.where(np.abs(MA_coefficients)>0.0)[0])

# Generate times:
t = np.arange(ndata)

# Generate white noise:
if sigma == 0.0:
        wn = np.zeros(len(t))
else:
        wn = np.random.normal(0,sigma,len(t))

# Generate flicker noise:
fn = Utils.ARMAgenerator(AR_coefficients,MA_coefficients,sigma_ARMA,len(t),burnin = len(t))
# Normalize it to have unitary variance:
fn = fn/np.sqrt(np.var(fn))

# Plot time-series and PSD:
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(12, 14))

# ------------------ Plot the time-series ------------------------------------
ax = plt.subplot(211)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Time')
plt.ylabel('Signal')

plt.plot(t,wn+fn,'-',color='black')

# ------------------ Plot the PSD --------------------------------------------
# Remove the plot frame lines. They are unnecessary chartjunk.
ax2 = plt.subplot(212)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

if sigma == 0.0:
    f,PSD = Utils.get_LS(t,wn+fn,np.ones(len(t)))
else:
    f,PSD = Utils.get_LS(t,wn+fn,np.ones(len(t))*sigma)

plt.xlabel('Frequency')
plt.ylabel('PSD')

plt.plot(f,PSD,'-',color='black')

# Plot the "real" PSD on top, obtained averaging lots of realizations of the process:
PSD_final = np.copy(PSD)
for i in range(nsims):
    if sigma == 0.0:
            wn_s = 0.0
    else:
            wn_s = np.random.normal(0,sigma,len(t))
    fn_s = Utils.ARMAgenerator(AR_coefficients,MA_coefficients,sigma_ARMA,len(t),burnin = len(t))
    fn_s = fn_s/np.sqrt(np.var(fn_s))
    if sigma == 0.0:
            f_s,PSD_s = Utils.get_LS(t,wn_s+fn_s,np.ones(len(t))*1.0)
    else:
            f_s,PSD_s = Utils.get_LS(t,wn_s+fn_s,np.ones(len(t))*sigma)
    PSD_final = PSD_final + PSD_s
PSD_final = PSD_final/(1. + nsims)
plt.plot(f,PSD_final,'-',color='red')

plt.show()

# Save data:
fout = open('simulated_data/arma_'+str(AR_order)+'_'+str(MA_order)+'_sigma_'+str(sigma)+'_sigma_ARMA_'+str(sigma_ARMA)+'.dat','w')
fout.write('# AR coeffs: ')
for c in AR_coefficients:
	fout.write(str(c)+'\t')
fout.write('\n')
fout.write('# MA coeffs: ')
for c in MA_coefficients:
        fout.write(str(c)+'\t')
fout.write('\n')
fout.write('# Time \t Signal \t Error\n')
for i in range(len(t)):
	fout.write(str(t[i])+'\t'+str(fn[i]+wn[i])+'\t'+str(sigma)+'\n')
fout.close()
