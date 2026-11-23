#!/usr/bin/env bash
# Reproduces the whole benchmark (30-min segments) from the public CTU-UHB data.
# Usage: bash run_all.sh [path/to/ctu-uhb-ctgdb/1.0.0]
set -euo pipefail

DATA=${1:-data/ctu-uhb-ctgdb/1.0.0}
OUT=outputs

# 1. Preprocessing (last 30 min; FOURIER, SPLINES and LLR keep oscillations up to 0.15 Hz)
python preprocessing/process_ctu_uhb.py --data_dir "$DATA" --out_dir $OUT/preprocessed_30 \
    --cut_min 30 --f_cut 0.15 --bandwidth_s 1.25

# 2. Features (linear, morphological, nonlinear and IBTF) for each pipeline
python features/build_features.py --in_dir $OUT/preprocessed_30 --out_dir $OUT/features_30

# 3. Repeated nested cross-validation of every pipeline (pH < 7.15)
python training/train_models.py --features 30=$OUT/features_30 --data_dir "$DATA" \
    --out_dir $OUT/results --ph_threshold 7.15

# 4. Statistical analysis and figures
python analysis/factor_analysis.py --results_dir $OUT/results --out_dir $OUT/figures
python analysis/plot_heatmap.py --results_dir $OUT/results --out_dir $OUT/figures
python analysis/plot_boxplots.py --results_dir $OUT/results --out_dir $OUT/figures --mean_sd
python analysis/plot_preprocessing.py --data_dir "$DATA" --f_cut 0.15 --bandwidth_s 1.25 --n_examples 1 \
    --out_dir $OUT/figures/preprocessing