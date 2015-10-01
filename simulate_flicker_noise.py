import numpy as np
import Utils

############## OPTIONS ##############
sigma = 0.0
#####################################

# Generate times:
t = np.arange(100)

# Generate white noise:
if sigma == 0.0:
        wn = np.zeros(len(t))
else:
        wn = np.random.normal(0,sigma,len(t))

# Generate flicker noise:
fn = Utils.FlickerGenerator(len(t),1.0)
# Normalize it to have unitary variance:
fn = fn/np.sqrt(np.var(fn))

# Plot time-series and PSD:
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.figure(figsize=(12, 14))

# ------------------ Plot the time-series ------------------------------------
ax = plt.subplot(211)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.xlabel('Time')
plt.ylabel('Signal')

plt.plot(t,wn+fn,'-',color='black')

# ------------------ Plot the PSD --------------------------------------------
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
nsims = 1000
PSD_final = np.copy(PSD)
for i in range(nsims):
        if sigma == 0.0:
                wn_s = 0.0
        else:
                wn_s = np.random.normal(0,sigma,len(t))
        fn_s = Utils.FlickerGenerator(len(t),1.0)
        fn_s = fn_s/np.sqrt(np.var(fn_s))
        if sigma == 0.0:
                f_s,PSD_s = Utils.get_LS(t,wn_s+fn_s,np.ones(len(t)))
        else:
                f_s,PSD_s = Utils.get_LS(t,wn_s+fn_s,np.ones(len(t))*sigma)
        PSD_final = PSD_final + PSD_s
PSD_final = PSD_final/(1. + nsims)
plt.plot(f,PSD_final,'-',color='red')


plt.show()

# Save data:
fout = open('simulated_data/fn_sigma_'+str(sigma)+'.dat','w')
fout.write('# Time \t Signal \t Error\n')
for i in range(len(t)):
	fout.write(str(t[i])+'\t'+str(fn[i]+wn[i])+'\t'+str(sigma)+'\n')
fout.close()
