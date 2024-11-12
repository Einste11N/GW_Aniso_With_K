# GW_Aniso_With_K

The main codes for evaluating the hyper-spherical Bessel functions and Transfer functions are in K0_flat, Km_open and K_p_closed, the stuctures of which are given below, where XXX stands for flat, open and close.

| **File names**            |  **Functions** |
|-----                      |-----|
| MyHyperSphericaljl.py     | Generating hyper-spherical Bessels |
| MyOtherFunctions.py       | Self-defined functions for numerical integral and differential |
| MySpherical_jl_test.ipynb | Debug files for MyHyperSphericaljl.py |
| Data_analysis.ipynb       | Debugs and Visualizations for hyper-spherical Bessels and some other quantities |
| Theta_l_XXX.py            | Evaluating the transfer functions |
| Theta_l_XXX.ipynb         | Notebook version of Theta_l_XXX.py |

The codes for performing the auto-correlation of GWB and its cross-correlations with CMB are provided in the dictionary Computation.

**NO DATA PROVIDED HERE**. The hyper-spherical Bessel functions and Transfer functions' files are too large to upload. If one wants to re-produced the result with different cosmological models, it is recommonded to go through the codes, change the parameter and run. One may need to modify the paths of data as well.