# What really matters for intrapartum fetal hypoxia detection?

Code of the paper *"What Really Matters for Intrapartum Fetal Hypoxia Detection? A Benchmark of
Machine Learning Pipelines"* (S. B. Alonso, C. M. Alaiz, Á. Fernández-Pascual).

The repository runs a systematic benchmark of machine learning pipelines for the detection of fetal
acidemia from the intrapartum fetal heart rate (FHR), on the public
[CTU-UHB database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/). Every pipeline is evaluated
under identical conditions (same outcome, same nested cross-validation partitions, same statistical
analysis), and the contribution of each factor to the performance is quantified.

| Factor | Levels |
|---|---|
| Preprocessing | `SOTA` (Cömert et al., polynomial detrending), `BASIC`, and three functional pipelines: `FOURIER`, `SPLINES`, `LLR` |
| Feature set (cumulative) | `M` (morphological), `M+L` (+ linear), `M+L+NL` (+ nonlinear), `M+L+NL+IBTF` (+ image-based time-frequency) |
| Classifier | SVM, LASSO, RIDGE, Random Forest, kNN, Decision Tree, multilayer perceptron with dropout (ANN) |

5 preprocessing pipelines × 4 feature sets × 7 classifiers = 140 pipelines, each evaluated with
5 × repeated nested cross-validation (5 outer folds, 3 inner folds).

## Repository structure

```
preprocessing/
    preprocess_fhr.py         the five preprocessing pipelines (SOTA, BASIC, FOURIER, SPLINES, LLR)
    process_ctu_uhb.py        applies them to the whole CTU-UHB database
features/
    morphological_features.py baseline, deceleration area (DCC) and acceleration area (ACC)
    linear_features.py        mean, std, STV, LTV, delta, interval index
    nonlinear_features.py     approximate and sample entropy, Lempel-Ziv complexity
    ibtf_features.py          GLCM texture of the spectrogram in four frequency bands
    clinical_features.py      optional clinical variables (not used in the paper)
    build_features.py         computes every feature for every preprocessing pipeline
training/
    models.py                 classifiers and hyperparameter grids (incl. the NumPy MLP with dropout)
    train_models.py           repeated nested cross-validation of every pipeline
analysis/
    common.py                 shared helpers (loading results, ANOVA, Friedman/Nemenyi)
    factor_analysis.py        variance decomposition (eta^2), Friedman tests, CD diagrams, interactions
    plot_heatmap.py           heatmap of the AUC of every pipeline
    plot_boxplots.py          AUC by level of each factor
    plot_preprocessing.py     figure of the five pipelines on one record
run_all.sh                    runs the whole benchmark
```

## Installation

```bash
conda env create -f environment.yml
conda activate ctg-hypoxia-benchmark
```

or `pip install -r requirements.txt` (Python ≥ 3.9).

## Data

Download CTU-UHB from PhysioNet (open access) into `data/`:

```bash
wget -r -N -c -np -nH --cut-dirs=1 -P data https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/
```

The scripts expect the folder with the `.hea`/`.dat` files, e.g. `data/ctu-uhb-ctgdb/1.0.0`.
The umbilical artery pH is read from the `.hea` headers.

## Reproducing the results

The whole benchmark runs with:

```bash
bash run_all.sh data/ctu-uhb-ctgdb/1.0.0
```

or step by step:

```bash
# 1. Preprocessing: last 30 min before delivery; FOURIER and SPLINES keep oscillations up to
#    0.15 Hz (541 Fourier and 543 B-spline basis functions for 30 min)
python preprocessing/process_ctu_uhb.py --data_dir data/ctu-uhb-ctgdb/1.0.0 \
    --out_dir outputs/preprocessed_30 --cut_min 30 --f_cut 0.15

# 2. Features for each preprocessing pipeline
python features/build_features.py --in_dir outputs/preprocessed_30 --out_dir outputs/features_30

# 3. Repeated nested cross-validation (label: pH < 7.15)
python training/train_models.py --features 30=outputs/features_30 \
    --data_dir data/ctu-uhb-ctgdb/1.0.0 --out_dir outputs/results

# 4. Statistical analysis and figures
python analysis/factor_analysis.py --results_dir outputs/results --out_dir outputs/figures
python analysis/plot_heatmap.py   --results_dir outputs/results --out_dir outputs/figures
python analysis/plot_boxplots.py  --results_dir outputs/results --out_dir outputs/figures --mean_sd
python analysis/plot_preprocessing.py --data_dir data/ctu-uhb-ctgdb/1.0.0 --f_cut 0.15
```

Training is resumable: if it is interrupted, running the same command again skips the
completed units. A quick test run: add `--seeds 0 --outer 3 --models LASSO --methods BASIC`
to step 3. `python training/train_models.py --show_grid` prints the pipeline and the
hyperparameter grid of every model.

### Outputs

| File | Content |
|---|---|
| `outputs/results/fold_results.csv` | AUC, AP, Brier score, balanced accuracy, sensitivity and specificity of every outer fold, with the selected hyperparameters |
| `outputs/results/predictions.csv` | out-of-fold predicted probability of every record |
| `outputs/results/summary.csv` | mean and SD across seeds of every pipeline |
| `outputs/figures/anova.csv`, `friedman.csv`, `posthoc_*.csv`, `ranks_*.csv` | statistical analysis |
| `outputs/figures/*.pdf` | figures (heatmap, boxplots, CD diagrams, eta^2, interactions) |

## Methodological notes

- **Outcome**: acidemia defined as umbilical artery pH < 7.15 (105 of 552 recordings).
- **No data leakage**: features are computed per record; imputation (median) and standardisation
  are fitted inside each training partition; hyperparameters are tuned only in the inner loop.
- **Class imbalance**: handled with class weights (`None` or `balanced`) searched as a
  hyperparameter, not with synthetic oversampling.
- **Paired comparisons**: for a given seed, every pipeline uses the same partitions.
- **Statistics**: cross-validation estimates on the same records are not independent, so the
  factorial ANOVA is used only as a variance decomposition (eta^2), and levels are compared
  with Friedman tests and Nemenyi critical-difference diagrams.

## Citation

The CTU-UHB database:
Chudáček V, Spilka J, Burša M, et al. *Open access intrapartum CTG database.*
BMC Pregnancy Childbirth. 2014;14:16.

## License

MIT (see `LICENSE`).
s