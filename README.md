# dPL-CH4

**Differentiable Parameter Learning for wetland methane (CH₄) spatial modeling**

dPL-CH4 couples a process-based methane formulation with machine learning. A spatialization network maps static environmental attributes to spatially varying base emission factors (\(k\)) and baseline temperature sensitivities (\(Q_{10,0}\)), while neural emulators learn nonlinear soil-water and vegetation constraints inside the process equations. The same codebase supports multi-site spatial cross-validation at flux towers and grid-scale global extrapolation.

## Project structure

- `DifferLand/`: JAX implementation of the spatial DifferLand / dPL-CH4 model
  - `model/`: DALEC-based carbon–water core and methane dynamic networks
  - `optimization/`: forward operators and loss terms
  - `util/`: multi-site data loading, spatial parameter nets, training, and static-feature normalization
- `drivers/`: example site NetCDF drivers, `site_coords.csv`, static-feature norm defaults, and an example CV fold JSON
- `tools/global_extrapolation/`: global Stage-2 extrapolation (driver build, forward, NetCDF writers)
- `tools/geo_data_process/`: meteorology packing and static-feature utilities used by global extrapolation
- `spatial_calibration.py`: calibration entry point (`--stage 1` / `--stage 2`)
- `run_stage1.sh` / `run_stage2.sh`: Stage-1 and Stage-2 training on bundled example sites
- `run_stage1_cv.sh` / `run_stage2_cv.sh`: fold-driven CV using `drivers/example_cv_folds.json`
- `run_global_stage2_extrapolation.sh`: global extrapolation launcher

## Two-stage training

1. **Stage 1 (carbon–water).** Train the spatially parameterized carbon–water model on multi-site drivers. Methane heads are inactive. Outputs include spatial weights used as the biophysical backbone for Stage 2.
2. **Stage 2 (methane).** Freeze Stage-1 weights and calibrate the methane module (\(k\), \(Q_{10,0}\), and gate emulators). Loss: \(L=-\lambda_{\mathrm{CH4}}\,\mathrm{NNSE}+L_{\mathrm{mean}}\).

Global runs apply Stage-2 parameters on a geographic grid with meteorology, static attributes, wetland extent, and a land mask supplied via `PATHS_ENV` / `MASK_NPY`.

## Setup

```bash
conda env create -f environment.yml
conda activate dPL-CH4
```

## Quick start

```bash
# Stage 1
bash run_stage1.sh

# Stage 2 (reads Stage-1 weights under output/stage1/)
bash run_stage2.sh
```

Fold-driven CV on the same example sites:

```bash
bash run_stage1_cv.sh
bash run_stage2_cv.sh
```

Override iteration count or Python if needed:

```bash
ITERATIONS=20 PYTHON=python bash run_stage1.sh
```

## Global extrapolation

```bash
PATHS_ENV=/path/to/paths.env \
MASK_NPY=/path/to/mask.npy \
STAGE2_ENS_ROOT=/path/to/stage2_ensemble \
SEEDS=42 START_YEAR=2001 END_YEAR=2001 \
bash run_global_stage2_extrapolation.sh
```

## Notes

- Bundled example sites: `BW-Gum`, `BW-Nxr`, `ID-Pag`.
- Defaults in `spatial_calibration.py` match these sites; replace drivers and coordinates for other networks.
- Training writes under `./output/` (gitignored).

## License

MIT. This package builds on DifferLand (Copyright 2023 Jianing Fang). See `LICENSE`.

## Citation

Please cite the accompanying dPL-CH4 manuscript when using this code.
