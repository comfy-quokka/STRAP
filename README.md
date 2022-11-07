# **ST**ructural **R**eliability **A**naylsis using **P**ython (STRAP)

STRAP is an package for structural reliability analysis.

STRAP supports 5 reliability analysis.
 - First Order Reliability Method (FORM)
 - Second Order Reliability Method (SORM)
   - Curvature fitting SORM
   - Compute failure probability for Breitung & Imporved Breitung formula
 - System Reliability
   - Matrix-based System Reliability (MSR)
   - Approximated by FORM & Dunnett-Sobel(DS) class
 - Monte Calro Simulation
 - Importance Sampling
   - Sample density ~ N( M = weighted sum of design points, S = S_sample ), weight = beta^(-m)
 
You have to declare some variables to use operator.py file.<br>
The variable declaration rule can be checked through the input_file.py file.<br>
User have to download Scipy, Numpy, and Sympy package to operate.<br>
 
If you have any questions, please send me an email.
