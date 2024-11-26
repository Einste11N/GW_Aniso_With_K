# GW_Aniso_With_K
Codes for arxiv:2410.17721.

The auto and cross correlation power spectrum results are shown in notebook 'Dl_and_Cross_Plots.ipynb', and data are provided in dictionary Dl_data. SNR analysis and estimations are shown in 'SNR_analysis.ipynb'. Main computation codes are in '/Codes'.

**Modules requirement** : numpy (basic computation), scipy (for interpolation), camb (for CMB data), pytorch (for speeding up with auto-parallel and GPU), matplotlib (for plotting), schnell (for SNR estimation). My environment is python3.8.19 with numpy=1.21.5, scipy=1.9.3, camb=1.5.4, pytorch=1.10.2, matplotlib=3.5.2, schnell=0.2.0. It should work well for higher version pythons for the codes are using just some basic functions.

**Note** : We made some change to the source code of schnell to evaluate inverse matrix of correlation matrix from LISA at low frequency. Since the correlation matrix $C$ for the three detectors of LISA will degenerate at low frequency, we simpliy take a cutoff, replacing $C$ at $f<=1e-3$ Hz with that at 1e-3 Hz.