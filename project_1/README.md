For the purpose of modeling a fabry-perot cavity and clipping of a Gaussian beam. 
Numpy, Scipy, Matplotlib, and Argparse are needed. \
All needed examples can be found here: 
# ODE Examples:
  ## Ringdown, no detuning
  python main.py ode --mode ringdown --T1 1e-3 --T2 1e-3 --Lrt 1e-4 --tmax 0.1 --npts 2000

  ## Ringdown, detuned
  python main.py ode --mode ringdown --method rk4 --delta 2e5 --tmax 0.1 --npts 200000

  ## Step-on, on resonance
  python main.py ode --mode stepon  --method all --tmax 5e-3 --npts 50000 --s0 1.0 --delta 0

  ## Step-on, detuned
  python main.py ode --mode stepon  --method all --tmax 1e-3 --npts 20000 --s0 1.0 --delta 1e5

# Integral Examples:
  python main.py integral --w 3e-3 --n 2000 --nsamp 25 --rule all \
  python main.py integral --w 3e-3 --n 500  --nsamp 40 --rule simpson
