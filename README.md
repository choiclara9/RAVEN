# RAVEN: Retinal Angiography and Vessel Ensemble Network

Code of team kirby for the [MICCAI 2026 GAVE2 Challenge](https://aistudio.baidu.com/competition/detail/1463/0/introduction):
*RAVEN – Retinal Angiography and Vessel Ensemble Network for CFP–FFA Segmentation and Biomarker Quantification*.

Jinwon Choi, Lynn Y. Kim — Ewha Womans University

| Final test (50 cases) | Task 1 | Task 2 | Task 3 | Overall |
|---|---|---|---|---|
| Score | 8.3681 | 8.4054 | 7.0022 | 7.8367 |

## Method

| Task | Input | Pipeline |
|---|---|---|
| 1. Artery/vein/vessel segmentation | CFP | Artery and vein from the public R2-V2 model (8-view TTA, not fine-tuned); vessel from a 5-fold RRWNet ensemble |
| 2. Artery/vein/vessel segmentation | CFP + early/late FFA | 3-fold CMRRWNet ensemble with topology-aware fine-tuning, endpoint-only vein bridging and additive vein-skeleton support |
| 3. Biomarkers (CRAE, CRVE, AVR, A/V density, A/V fractal dimension) | CFP + FFA | 3-model CMRRWNet ensemble + SegFormer optic-disc mask, fixed global calibration |

## Repository

```text
model.py, utils.py, preprocessing.py     RRWNet / CMRRWNet and helpers
get_predictions.py                       inference of one checkpoint
ensemble_predictions.py                  equal-weight fold averaging
build_*_submission.py                    Task 2 post-processing, Task 3 calibration
get_biomarker_kaggle.py                  Task 3 biomarker extraction
pipeline/                                end-to-end pipeline (raw data -> kirby.zip), manifests
train/, scripts/                         training code and commands
docker/                                  Dockerfiles, entrypoints, weight download scripts
```

The inference code (root scripts and `pipeline/`) is the code used in the Docker images.

## Data

`DATA_DIR` is the official GAVE2 data (not redistributed):

```text
DATA_DIR/training/biomarker/g_001.txt ... g_050.txt
DATA_DIR/validation/{images,masks,FFA_A,FFA_AV}/g_051.png ... g_100.png
```

## Weights

- Our checkpoints (11 files, 187 MB): [gave2_kirby_weights.zip](https://github.com/choiclara9/RAVEN/releases/download/v1.0-gave2-final/gave2_kirby_weights.zip)
- R2-V2 ([source](https://github.com/j-morano/R2-V2)) and SegFormer optic disc/cup ([Hugging Face](https://huggingface.co/pamixsun/segformer_for_optic_disc_cup_segmentation)) are public models, fetched by pinned scripts:

```bash
unzip gave2_kirby_weights.zip               # -> weights/{task1,task2,task3}
bash docker/fetch_r2v2.sh weights           # -> weights/external/r2v2
bash docker/fetch_optic_disc.sh weights     # -> weights/external/optic_disc
python pipeline/pipeline.py verify-weights --weights-root weights
```

## Inference (Docker)

Requires Docker, NVIDIA Container Toolkit and an NVIDIA GPU. Two images are used because our models need PyTorch 2.2 / CUDA 12.1 and R2-V2 needs PyTorch 2.8 / CUDA 12.8. The `-weights-` images of [`choiclara9/gave2-kirby`](https://hub.docker.com/r/choiclara9/gave2-kirby) already contain all weights.

```bash
DATA_DIR=/path/to/GAVE2
OUT_DIR=$PWD/kirby_output
mkdir -p "$OUT_DIR/r2v2" "$OUT_DIR/main"

docker run --rm --gpus all -v "$DATA_DIR:/data:ro" -v "$OUT_DIR/r2v2:/output" \
  choiclara9/gave2-kirby:prelim-7.95236-weights-r2v2

docker run --rm --gpus all -v "$DATA_DIR:/data:ro" \
  -v "$OUT_DIR/r2v2/r2v2:/external/r2v2:ro" -v "$OUT_DIR/main:/output" \
  choiclara9/gave2-kirby:prelim-7.95236-weights-main run-all
```

The result is `$OUT_DIR/main/kirby.zip` (`Task1/*.png`, `Task2/*.png` with R = artery, G = vessel, B = vein, and `Task3/*.txt`). On the preliminary data its SHA-256 is

```text
4c19a97503668ef003c992edcbf76ade644c40903b683394347c0214db51306c
```

The hash above and the `prelim-7.95236` image tags refer to the preliminary round (overall 7.9524). The final-round score (7.8367, table above) was obtained with the same weights and code, applied as described below.

SHA-256 of the submitted final-round archive (100 cases, `Task1/`, `Task2/`, `Task3/` with `g_001` … `g_100`):

```text
bc65562dde89faa8edaeed8ea2017253c1a25b6a3aad730d6f35d38dc5f269a5
```

### Reproducing the final-round result (technical report, Section 3.1)

The final-round input has 100 cases (`g_001` … `g_100`): the 50 preliminary cases (shuffled in among the final-round numbering, 25 in each half) and the 50 new final-round cases. The pipeline expects exactly 50 cases named `g_051` … `g_100`, and the Task 3 calibration ranks the 50 cases of one run, so the 100 cases were processed as **two batches of 50**: `g_001` … `g_050` and `g_051` … `g_100`. Each batch needs both Docker runs above (the R2-V2 image first, then the main image):

1. Put the 50 cases of one batch into `DATA_DIR/validation/{images,masks,FFA_A,FFA_AV}`, renamed to `g_051.png` … `g_100.png` (`DATA_DIR/training/biomarker` must still contain the 50 training labels). For the first batch, `g_001` … `g_050` become `g_051` … `g_100`; the second batch already has these names.
2. Run the two `docker run` commands above, using a separate `OUT_DIR` per batch and adding `--allow-hash-mismatch` to the `run-all` command (the output hash only matches for the preliminary data).
3. For the first batch, rename the files in `Task1/`, `Task2/` and `Task3/` of `kirby.zip` back to `g_001` … `g_050`.
4. Merge both batches into one archive with 300 entries.

Do not change the batch composition: Task 3 values depend on the rank of each case within its batch.

Without Docker (Python 3.10, `pip install -r requirements.txt`, PyTorch 2.2.0), R2-V2 has to be run first in a separate PyTorch 2.8 environment (`pip install -r docker/requirements-r2v2.txt`):

```bash
python weights/external/r2v2/source/infer.py --cfp_path DATA_DIR/validation/images \
  --masks_path DATA_DIR/validation/masks --model_type av --weights_path weights/external/r2v2 \
  --save_path R2V2_OUT --tta --use-gave-format

python pipeline/pipeline.py run-all --data-root DATA_DIR --weights-root weights \
  --work-root work --output out/kirby.zip --r2-av-source R2V2_OUT/av
```

## Training

Place the official training set under `data/training/{images,masks,av,FFA_A,FFA_AV}` and use a CUDA GPU.

```bash
bash scripts/train_task1.sh 0      # RRWNet, folds 0-4
bash scripts/train_task2.sh 0      # CMRRWNet baseline, folds 0-2
bash scripts/finetune_task2.sh 0   # topology-aware fine-tuning, RRLoss(OfficialScoreLoss), lr 1e-5
```

Settings: Adam (no weight decay), lr 1e-4, mixed precision, batch size 1, seed 77, base width 16, early stopping on validation loss (patience 100). Task 1: RRWNet, 3 input channels, 5 refinement iterations, 5-fold split. Task 2: CMRRWNet, 5 input channels, 1 refinement iteration, 4-fold split. Checkpoints are written to `__training/`.

## License

MIT (see `LICENSE`). Built on [CMRRWNet](https://github.com/Peng2004/CMRRWNet) and [RRWNet](https://github.com/j-morano/rrwnet).
