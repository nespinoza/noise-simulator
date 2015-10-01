# noise-simulator

This code generates indexed time-series that can come from an ARMA or a 1/f 
process plus white noise. The code will both save the time-series and plot it. This last plot, 
comes with a PSD estimation of both the time-series and the "real", "underlying" 
process which is obtained by simulating lots of simulations and averaging the obtained 
PSDs.

The ARMA and flicker noises are automatically normalized to have unitary variance; the white 
noise part is added on top of that.

DEPENDENCIES
------------

This code makes use of four important python libraries:

    + Numpy.
    + Scipy.
    + Matplotlib.
    + AstroML

All of them are open source and can be easily installed in any machine. 

USAGE
------------

Edit the parameters and simply run the simulate_arma_noise.py or the 
simulate_flicker_noise.py to simulate the noises. Simulations will be 
saved to the simulated_data folder.
