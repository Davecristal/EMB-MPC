# EMB-MPC: Online Evolving Model-Bank Model Predictive Control for Friction-Adaptive Autonomous Racing

This repository provides the simulation code used for EMB-MPC.

## Environment

The experiments were developed with:

- Python 3.9
- JAX 0.4.30
- CUDA 12.2
- CasADi 3.7.2
- IPOPT through CasADi

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Use the CUDA/JAX wheel matching your machine.

## Run

If you want to reproduce the experiments, use the following commands:

EMB-MPC:

```bash
python -m embmpc.mpc.run_nmpc_orca_embmpc --help
python -m embmpc.mpc.run_nmpc_orca_embmpc_avg_runs --help
```

## Interpretation Notes

