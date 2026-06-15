# EMB-MPC: Online Evolving Model-Bank Model Predictive Control for Friction-Adaptive Autonomous Racing

This repository provides the simulation code used for EMB-MPC, LLA-MPC fixed-bank baselines, APACRace, and the idealized informed Oracle reference.

The release focuses on ETHZ and ETHZMobil friction-varying autonomous racing experiments. Historical training scripts, CARLA/F110 variants, old plotting scripts, cached files, and unused safe-set/value-function experiments have been removed from the paper reproduction tree.

## Environment

The paper experiments were developed with:

- Python 3.9
- JAX 0.4.30
- CUDA 12.2
- CasADi 3.7.2
- IPOPT through CasADi

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Use the CUDA/JAX wheel matching your machine. For CPU-only smoke tests, install the CPU JAX wheel.

## Entry Points

EMB-MPC:

```bash
python -m embmpc.mpc.run_nmpc_orca_embmpc_unknown_detector --help
python -m embmpc.mpc.run_nmpc_orca_embmpc_avg_runs --help
```

Baselines and reference:

```bash
python -m embmpc.mpc.run_nmpc_orca_llampc_rt --help
python -m embmpc.mpc.run_nmpc_orca_llampc_nrt --help
python -m embmpc.mpc.run_nmpc_apacrace --help
python -m embmpc.mpc.run_nmpc_oracle --help
```

## Reproduction Commands

Experiment 1, EMB-MPC single-run diagnostics on ETHZMobil:

```bash
python -m embmpc.mpc.run_nmpc_orca_embmpc_unknown_detector --n_models 20000 --seed 42 --no_video --no_plots --output_file results/embmpc_ethzmobil_n20000_seed42.json
```

Experiment 2, EMB-MPC 10-run average on ETHZ:

```bash
python -m embmpc.mpc.run_nmpc_orca_embmpc_avg_runs --n_runs 10 --n_models 5000 --output_file results/embmpc_ethz_n5000_10runs.json
python -m embmpc.mpc.run_nmpc_orca_embmpc_avg_runs --n_runs 10 --n_models 10000 --output_file results/embmpc_ethz_n10000_10runs.json
python -m embmpc.mpc.run_nmpc_orca_embmpc_avg_runs --n_runs 10 --n_models 20000 --output_file results/embmpc_ethz_n20000_10runs.json
```

Experiment 3, baselines and idealized reference:

```bash
python -m embmpc.mpc.run_nmpc_orca_llampc_rt --n_models 5000
python -m embmpc.mpc.run_nmpc_orca_llampc_nrt --n_models 5000
python -m embmpc.mpc.run_nmpc_apacrace
python -m embmpc.mpc.run_nmpc_oracle
```

The six friction-varying benchmark scenarios and track-level bookkeeping are summarized in `configs/ethz.yaml` and `configs/ethzmobil.yaml`. The original experiment scripts keep the corresponding scenario timing and friction-update logic near each script's `update_friction` block.

## Interpretation Notes

- Oracle is an idealized informed reference, not a practical deployable controller.
- EMB-MPC, LLA-MPC, and APACRace do not access the imposed benchmark profile or the transition time.
- Runtime numbers should be interpreted as relative comparisons for the current Python/JAX/CasADi prototype.
- Generated figures, videos, JSON summaries, NumPy arrays, and logs are ignored by Git.
