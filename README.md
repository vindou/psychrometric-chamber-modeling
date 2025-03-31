# Psychrometric Chamber Modeling
This repository contains all the code written by [Gilbert Chang](https://gilcha.com/) as a part of research work done in collaboration with [Jie Cai](https://engineering.purdue.edu/ME/People/ptProfile?resource_id=71762&group_id=11989) at Herrick Laboratories. 

# Running Scripts
## MATLAB
The MATLAB scripts found in /matlab/ were written in Matlab_R2024b, making extensive use of [CVX](https://github.com/cvxr/CVX) and [the MATLAB Control System Toolbox](https://github.com/cvxr/CVX). Installation of these prerequisites is recommended before attempting to run the scripts. 
## Python
Clone the repository or simply download the files from this GitHub page to obtain the scripts.
```
git clone https://github.com/vindou/psychrometric-chamber-modeling
```
Use your terminal of choice to enter the directory containing the repository. The Python scripts were tested with Python 3.12 in mind, so setting up a virtual environment with the same version is recommended but optional. Run the following to get all the dependencies.
```
pip install -r requirements.txt
```
After dependencies have been installed, run the `python [script file name].py` to run any script in the repository. Detailed documentation regarding the function of each script will come soon.
