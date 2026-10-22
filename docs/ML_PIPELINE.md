# ML Pipeline

## Pipeline

~~~text
Food-101
    → selected classes
    → train / validation / official test
    → image transforms
    → PyTorch Dataset
    → DataLoader
    → Phase 3 model training / validation
~~~

The final step is implemented as a Phase 3 consumer. Phase 2 stops after producing
verified image batches; its dataset and preprocessing implementation is unchanged.

## Loading and filtering

ml/src/download_dataset.py prepares the dataset under ml/data/raw/ using
torchvision.datasets.Food101. ml/src/dataset.py loads both official splits and
validates metadata, class agreement, image paths, file existence, and official split
isolation.

The selected classes are filtered by their Food-101 names. Food101Subset stores
indices into the original Food101 object, so no image files are duplicated. Each
subset exposes classes, class_to_idx, indices, targets, and image_paths.

The shared application label order is:

~~~text
0 pizza
1 hamburger
2 sushi
3 steak
4 ice_cream
5 donuts
6 omelette
7 french_fries
8 chicken_wings
9 spaghetti_bolognese
~~~

## Split behavior

The default DataConfig is:

~~~text
image_size=224
batch_size=32
num_workers=0
random_seed=42
validation_fraction=0.2
~~~

Each selected official training class is split into 600 training and 150 validation
samples. The selected official test samples remain all 250 per class. Paths are
sorted before deterministic per-class shuffling. SHA-256 image-content groups are
kept together so identical images cannot cross a split boundary.

## Transforms

ml/src/preprocessing.py returns separate transform pipelines.

Training transform:

1. RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))
2. RandomHorizontalFlip(p=0.5)
3. ToTensor()
4. ImageNet normalization

Evaluation transform:

1. Resize shorter edge to 256 with bilinear interpolation and antialiasing
2. CenterCrop(224)
3. ToTensor()
4. ImageNet normalization

The normalization constants are mean (0.485, 0.456, 0.406) and standard deviation
(0.229, 0.224, 0.225). Images are converted to RGB before transformation.
Validation and test transforms contain no random operations.

## Dataset and DataLoader behavior

Each dataset returns an image and label:

- image: CPU torch.float32 tensor with shape [3, 224, 224] by default;
- label: CPU torch.int64 scalar using application labels 0–9.

Each DataLoader returns a batch:

- images: [batch_size, 3, 224, 224];
- labels: [batch_size].

The training loader shuffles. Validation and test loaders preserve deterministic
dataset order. All loaders use drop_last=False and pin_memory=False. The final
batch may be smaller than the configured batch size.

create_data_pipeline seeds Python, NumPy, and PyTorch generators and returns the
three datasets and three loaders. create_dataloaders supports configurable worker
counts and uses spawn when workers are enabled. Keep the if __name__ == "__main__"
guard in scripts that create loaders on macOS.

## Configuration

DataConfig accepts:

- root: Food-101 root directory;
- classes: ordered class names;
- image_size: positive output size;
- batch_size: positive batch size;
- num_workers: zero or more workers;
- random_seed: deterministic split and loader seed;
- validation_fraction: fraction of each selected training class.

The CLI scripts expose the same root, image-size, batch-size, worker, seed, and
validation-fraction options. The verification CLI intentionally requires the agreed
ten-class development configuration.

## Verification procedure

verify_pipeline.py verifies:

- selected classes, contiguous labels, and class mappings;
- complete train/validation partition and untouched official test selection;
- deterministic splits for repeated seed 42;
- unchanged test membership when the seed changes;
- path overlap and identical image bytes across splits;
- deterministic evaluation transforms;
- CPU tensor device, shapes, dtypes, finite values, and normalized bounds;
- first-batch reproducibility;
- complete iteration through all selected images with --all-batches.

The verified command was:

~~~bash
python ml/src/verify_pipeline.py --all-batches
~~~

It decoded and transformed 10,000 selected images and produced batches with shape
[32, 3, 224, 224] and labels [32]. A second real-data run with --num-workers 2
also passed. JSON reports are written under the ignored ml/data/processed/ directory.

## macOS and device boundary

Phase 2 does not select MPS, CUDA, or CPU for model execution. It creates CPU
tensors so Phase 3 can choose an appropriate device, typically mps, then cuda,
then cpu according to its own runtime policy. Zero workers are the default for
simple macOS behavior; positive workers are supported through spawn and were
verified with two workers.

## Phase 3 training

`model.py` creates torchvision EfficientNet-B0 with explicitly pinned
`EfficientNet_B0_Weights.IMAGENET1K_V1` initialization and replaces its final linear
layer with ten outputs. The class count/order comes from the Phase 2 development
configuration; no second class list is defined. All backbone and classifier
parameters are trainable. Dropout remains 0.2. Phase 2 transforms are reused rather
than replaced by a weights-specific transform pipeline.

`runtime.py` exposes `select_device("auto" | "mps" | "cuda" | "cpu")`. Automatic
selection prefers MPS, then CUDA, then CPU; an unavailable explicit request fails
clearly. The training loop simply moves images/labels to the selected device.
`seed_training` reuses Phase 2 Python/NumPy/PyTorch seeding, which also seeds
accelerators, and disables cuDNN benchmarking while requesting deterministic cuDNN
behavior. Seeds do not guarantee bitwise equality across backends or library versions.

`train.py` calls the existing `create_data_pipeline(DataConfig(...))`, then passes
only its train and validation loaders to the epoch functions in `training.py`.
The factory still creates test split metadata/index views and validates official
paths, as before. No test sample is passed to the model or used for any development
decision. Phase 3 does not download Food-101 automatically.

Default training settings:

| Setting | Value |
| --- | --- |
| Epochs | 5 |
| Batch size | 16 (CLI override; Phase 2 default remains 32) |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW, all layers |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Workers | 0 |
| Seed | 42 |
| Input / validation fraction | Existing 224×224 / 0.2 |
| Device | auto (MPS → CUDA → CPU) |
| Pretrained | true |
| Best checkpoint | ml/artifacts/efficientnet_b0_best.pt |

The completed run produced the following train/validation losses and accuracies:

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0329 | 70.73% | 0.4178 | 87.47% |
| 2 | 0.4081 | 87.32% | 0.3315 | 89.87% |
| 3 | 0.2817 | 91.47% | 0.3185 | 90.87% |
| 4 | 0.1908 | 93.83% | 0.2781 | **92.00% (best)** |
| 5 | 0.1413 | 95.55% | 0.3019 | 91.53% |

Checkpoint metadata confirms epoch 4, 92.00% validation accuracy, MPS, seed 42,
ImageNet `IMAGENET1K_V1` provenance, and the existing class order. Training is
complete; the best epoch-4 checkpoint is used for final test evaluation and inference.

The training CLI can start a fresh run (do not run it to reproduce the completed
experiment unless a new training experiment is intended):

~~~bash
python ml/src/train.py --epochs 5 --batch-size 16 --lr 0.0001
python -m ml.src.train --help
~~~

The CLI also accepts `--root`, `--weight-decay`, `--num-workers`, `--seed`,
`--device`, `--checkpoint`, and `--no-pretrained`. Changing the seed deliberately
changes the development split through Phase 2's own logic; the default remains 42.
Use a different checkpoint path to retain multiple runs: each invocation starts
fresh and replaces that path whenever its own validation accuracy improves.

The model factory announces whether pretrained weights are cached or need a
download. Torchvision uses its normal torch hub cache, configurable via `TORCH_HOME`.
A network/cache failure stops the run with context; it never silently falls back to
random weights. `--no-pretrained` is an explicit random-initialization option.

Each training batch performs zero-grad, forward, cross-entropy loss, backward, and
AdamW step. Loss/accuracy are weighted by sample count, including partial final
batches. Validation runs after every epoch using `model.eval()` and
`torch.no_grad()`, without an optimizer or parameter/buffer updates. Nonfinite loss
stops the run. Epoch-level logs report train/validation loss and accuracy.

## Checkpoint saving and loading

Strictly improved validation accuracy saves a checkpoint; ties retain the earlier
epoch. `checkpoint.py` writes a temporary file beside the destination and replaces
the destination atomically. `ml/artifacts/` and checkpoint extensions are ignored.

Checkpoints contain a format version, model name/configuration, ImageNet weight
provenance, model and optimizer state, ordered classes and class-to-label mapping,
data/split/seed settings, descriptions from the actual Phase 2 transforms, training
configuration, best epoch and metrics, history through that epoch, and library/device
information. They represent the best model, not the final epoch of every run.

~~~python
from pathlib import Path
from ml.src.checkpoint import load_checkpoint
from ml.src.runtime import select_device

model, metadata = load_checkpoint(
    Path("ml/artifacts/efficientnet_b0_best.pt"), device=select_device(),
)
classes = metadata["classes"]  # logits index -> existing Phase 2 class name
~~~

Loading uses `torch.load(..., map_location="cpu", weights_only=True)`, verifies
format/model identity, class order/mapping, classifier size, preprocessing, and key
training metadata, then strictly loads model weights. Reconstruction uses
`pretrained=False` even when the saved provenance is ImageNet, so loading never
downloads weights. The returned model is on the requested device in eval mode.
Optimizer state is included, but exact interrupted-run resumption is not implemented:
RNG/loader position and subsequent training history are not restored.

## Phase 3 smoke verification

~~~bash
python ml/src/verify_training.py
~~~

This defaults to CPU, batch size 2, workers 0, seed 42, and random weights. It uses
one real Phase 2 training batch for a single backward/AdamW step and one real
validation batch for repeatability, mode, and state-preservation checks. It verifies
the ten-output classifier, finite `[2, 10]` logits/loss, backbone/head updates,
checkpoint state/logits/optimizer/class-mapping round-trip, offline loading, and
rejection of incompatible checkpoints. Temporary checkpoints are removed afterward.
The JSON report is ignored at `ml/data/processed/phase3_verification.json`.

Additional controlled checks cover device selection/overrides/unavailable-device
errors, seeded generators, unequal batch metric aggregation, pretrained enum/error
handling, and best-checkpoint selection including ties. These do not execute GPU
kernels or fetch pretrained weights. `--device mps` or `--device cuda` runs the real
model checks on that device when available; `--pretrained` explicitly allows a real
ImageNet download for verification. `--num-workers 2` exercises the existing spawn
loader path without changing it.

For Phase 2 regression checks, use its existing `verify_pipeline.py --all-batches`.
That verifies all selected images, including test images, solely for dataset and
transform integrity; it does not compute model performance. Phase 3 smoke checks
do not evaluate test samples.

## Final held-out test evaluation

`evaluate.py` loads and metadata-validates the saved checkpoint using the existing
loader. The evaluation root, seed, class order, image size, and validation fraction
come from that checkpoint's `DataConfig`; batch size and worker count can be changed
without changing split membership. It reuses `create_data_pipeline` and its existing
deterministic test transform. Before any image is decoded, it checks the official
source split, checkpoint class mapping/transform, exact selected official indices,
unique paths, 250 samples per class, and a sequential loader that drops no samples.
After the single test pass it again requires 2,500 total predictions and 250 per
class.

The report contains summed CrossEntropy loss divided by the total number of samples,
overall accuracy, sample count, per-class sample/correct counts and accuracy, and a
10×10 numeric confusion matrix. Matrix rows are true classes, columns are predicted
classes; row/column indices follow the saved class order. Results print to the
terminal and a JSON report is saved under ignored `ml/artifacts/` by default.
The report records the checkpoint SHA-256, epoch, and validation accuracy. The
evaluator checks that the checkpoint hash stays unchanged and does not create an
optimizer or save model state. Existing output files and checkpoint aliases are
refused to protect the model artifact.

Exact command for the reviewed final evaluation:

~~~bash
python ml/src/evaluate.py --checkpoint ml/artifacts/efficientnet_b0_best.pt --device mps --batch-size 16 --num-workers 0 --report ml/artifacts/phase3_final_test.json
~~~

The final evaluation completed on Apple MPS. Its report is
`ml/artifacts/phase3_final_test.json`, which records the checkpoint hash, epoch 4,
validation accuracy, metrics, and confusion matrix. Results were:

| Measure | Final held-out result |
| --- | ---: |
| CrossEntropy loss | 0.176723 |
| Accuracy | 94.48% |
| Samples | 2,500 |

| Food class | Correct / total | Accuracy |
| --- | ---: | ---: |
| pizza | 240 / 250 | 96.00% |
| hamburger | 231 / 250 | 92.40% |
| sushi | 231 / 250 | 92.40% |
| steak | 225 / 250 | 90.00% |
| ice_cream | 235 / 250 | 94.00% |
| donuts | 238 / 250 | 95.20% |
| omelette | 231 / 250 | 92.40% |
| french_fries | 246 / 250 | 98.40% |
| chicken_wings | 242 / 250 | 96.80% |
| spaghetti_bolognese | 243 / 250 | 97.20% |

These are the final held-out results. Do not use them for further model or
hyperparameter tuning. The command above describes the completed run; the report
path must be new if evaluation is run again.

Run synthetic-only verification (it loads the saved checkpoint for synthetic
tensors, but never loads Food-101 or consumes official images):

~~~bash
python ml/src/verify_evaluation.py
~~~

Six checks cover metric aggregation and confusion orientation, malformed inputs,
class/mapping/transform/split/sampler validation, a full 2,500-sample synthetic
loader pass, output-path protections, and real-checkpoint immutability with synthetic
images. It asserts eval/no-grad, unchanged parameters/buffers/gradients, and unchanged
checkpoint bytes. It runs no training steps and blocks Food-101 loading.

## Single-image prediction

`inference.py` provides `FoodImagePredictor.from_checkpoint(...)` and reusable
`.predict(image_path, top_k=3)`. The checkpoint loader validates the model, ordered
classes, and preprocessing metadata and restores weights without ImageNet downloads.
The predictor selects `auto`, `mps`, `cuda`, or `cpu` through `runtime.py`; it derives
both class-name/index directions from the checkpoint's `class_to_idx`.

Input images are checked for file existence and successful decoding, converted to
RGB, and transformed with the existing Phase 2 deterministic evaluation transform.
The checkpoint's input size must be 224×224. `build_transforms` supplies resize,
center crop, tensor conversion, and ImageNet normalization; no preprocessing
constants or steps are duplicated in the inference implementation.

For each call the model enters eval mode and a no-grad context. A single-item batch
produces ten logits, then softmax probabilities. The result includes the mapped
predicted name, class index, confidence in `[0, 1]`, and the requested descending
top-k list. The operation creates no optimizer, updates no model state, and never
writes to the checkpoint.

CLI example:

~~~bash
python ml/src/predict.py path/to/image.jpg --checkpoint ml/artifacts/efficientnet_b0_best.pt --device mps --top-k 3
~~~

The CLI prints the predicted class, confidence as a percentage, and top predictions.
PIL-readable formats such as JPEG, PNG, and WebP are accepted by the installed PIL
decoder. Missing, corrupt, and unreadable paths fail with a clear error. The reusable
API returns the same prediction data as a dictionary.

Run inference verification:

~~~bash
python ml/src/verify_inference.py
~~~

The verifier loads the saved best checkpoint and predicts on temporary synthetic
RGBA and grayscale PNGs. It checks RGB conversion, the `[1, 3, 224, 224]` forward
input, finite ten-class probabilities summing to one, checkpoint-derived names and
indices, descending top-k output, CLI formatting, eval/no-grad behavior, unchanged
weights/buffers/gradients, and unchanged checkpoint bytes. It uses no dataset images,
does no metric evaluation, and takes no optimizer steps.

The classifier recognizes only the configured ten classes and has no explicit
unknown or out-of-distribution class. Phase 3 is COMPLETE. Its 94.48% held-out
result is classification accuracy only, not nutrition or calorie-estimation accuracy;
it remains frozen and was not rerun during Phase 4 or Phase 5.

Phase 4 adds a separate USDA FDC lookup and portion calculator for the existing ten
labels. It does not change checkpoint classes, model code, confidence, or inference.
See `docs/NUTRITION.md` for sources, default portions, calculations, and limitations.
Phase 5 has added a separate 13-class classifier and image-to-nutrition integration;
mobile application flow, editable portions, logging, goals, and dashboards remain
planned.

## Phase 5 — regional 13-class model and integration (COMPLETE)

Phase 5 preserves the Phase 3 classifier and checkpoint. Its class order is the
original ten labels followed by `borek`, `kebap`, and `sarma`. The source label
`yaprak_sarma` is mapped explicitly to `sarma`. See `docs/DATASET.md` for the pinned
TurkishFoods-25 revision, source split rules, and manifest format.

`download_phase5_dataset.py` exports only the three selected classes to ignored
local data. `phase5_data.py` combines them with the established Food-101 train and
validation subsets and official test subset. It preserves TurkishFoods-25 train,
eval, and test, verifies manifests and hashes, rejects content leakage across local
splits, and checks content overlap between the two datasets. Validation drives model
selection. The independent mixed test split is used only by `evaluate_phase5.py`.

Training creates EfficientNet-B0 with a fresh 13-output head, then copies only
shape-compatible `features.*` weights from the saved Phase 3 model. No Phase 3
classifier weights are loaded. All layers are fine-tuned with the existing 224×224
transforms and AdamW settings. The model trains on both datasets. Inverse-frequency
class weights `N / (13 × n_class)` normalized to mean 1 address actual class
imbalance without duplicating images. The separate checkpoint is
`ml/artifacts/efficientnet_b0_phase5_best.pt`; the result report is
`ml/artifacts/phase5_mixed_test.json`. Neither overwrites Phase 3 artifacts.

Run after dataset preparation:

~~~bash
python ml/src/train_phase5.py --epochs 5 --batch-size 16 --device auto
python ml/src/evaluate_phase5.py --device auto
python -m ml.src.phase5_pipeline path/to/meal.jpg
~~~

The five-epoch MPS run selected epoch 5 at 92.24% validation accuracy. Epoch-level
metrics were:

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3762 | 89.21% | 0.2975 | 91.82% |
| 2 | 0.1465 | 95.72% | 0.3026 | 91.35% |
| 3 | 0.1197 | 96.45% | 0.3116 | 91.76% |
| 4 | 0.0942 | 96.95% | 0.3107 | 91.59% |
| 5 | 0.0722 | 97.80% | 0.2970 | **92.24% (best)** |

The independent Phase 5 mixed test run evaluated all 2,500 official Food-101 test
images and 219 TurkishFoods-25 test images (2,719 total). Loss was 0.17226 and
overall accuracy was 95.66%. Per-class results:

| Class | Correct / total | Accuracy |
| --- | ---: | ---: |
| pizza | 245 / 250 | 98.00% |
| hamburger | 231 / 250 | 92.40% |
| sushi | 235 / 250 | 94.00% |
| steak | 230 / 250 | 92.00% |
| ice_cream | 234 / 250 | 93.60% |
| donuts | 245 / 250 | 98.00% |
| omelette | 230 / 250 | 92.00% |
| french_fries | 245 / 250 | 98.00% |
| chicken_wings | 245 / 250 | 98.00% |
| spaghetti_bolognese | 247 / 250 | 98.80% |
| borek | 76 / 78 | 97.44% |
| kebap | 80 / 82 | 97.56% |
| sarma | 58 / 59 | 98.31% |

This is a mixed-dataset classification result, not nutrition-estimation accuracy.
Phase 3's frozen 94.48% remains its historical ten-class Food-101 classification
accuracy. Phase 5 results did not guide Phase 3 selection. The report, including
confusion matrix and dataset counts, is `ml/artifacts/phase5_mixed_test.json`.

## Phase 5 image-to-nutrition behavior

The reusable `ImageNutritionPipeline` loads and validates the separate 13-class
checkpoint, applies the established deterministic image transform, returns a
canonical label and confidence, and performs exact Phase 4 catalog lookup. The
primary nutrition result uses the configured default serving. Confidence is
informational and is never multiplied into nutrients; no portion weight is inferred
from an image. The model has no unknown class, so unrelated images may still be
forced into one of the 13 supported labels. See `docs/NUTRITION.md` for the regional
nutrition record limitations.

## MobileNetV3 / TensorFlow Lite deployment model (Phase 7 groundwork)

The assignment requires a MobileNet-family model and a TFLite deliverable. The
earlier PyTorch EfficientNet-B0 checkpoints and their results remain historical
baselines; no Phase 3 or Phase 5 model/evaluation artifact was overwritten. The
current mobile model is a separate Keras MobileNetV3 Small with ImageNet pretrained
backbone weights, all backbone layers fine-tuned, and a fresh dropout-0.2 plus
13-output softmax head. EfficientNet weights were not copied because the model
architectures differ.

MobileNetV3 Small was selected for a mobile-size/latency tradeoff. It is much
smaller than the baseline checkpoint and the exported dynamically weight-optimized
TFLite file is about 1.1 MB. Its measured classification accuracy is lower than the
EfficientNet-B0 baseline, so those results remain distinct and should be considered
when choosing the final product model.

### Phase 5 data adapter and preprocessing

`tf_phase5_data.py` adapts the existing validated Phase 5 datasets rather than
creating new splits. It retains the Food-101 training-derived train/validation
membership, official Food-101 test membership, pinned TurkishFoods-25 train/eval/test
separation, `yaprak_sarma` → `sarma` mapping, and existing path/content leakage
checks. The exact 13-class order is:

~~~text
0 pizza
1 hamburger
2 sushi
3 steak
4 ice_cream
5 donuts
6 omelette
7 french_fries
8 chicken_wings
9 spaghetti_bolognese
10 borek
11 kebap
12 sarma
~~~

The Phase 5 mixed data used here contains 7,684 training images, 1,700 validation
images, and 2,719 held-out test images. Per-class split counts are preserved in the
training metadata and test report. The existing inverse-frequency class weights are
used without duplicating samples. The adapter obtains existing 224×224 normalized
samples, reverses the Phase 2 ImageNet normalization to raw RGB [0,255], and then
lets Keras MobileNetV3's built-in preprocessing rescale inputs to [-1,1].

Mobile and Python inference use the same deterministic evaluation geometry: convert
to RGB, resize the shortest edge to 256 with bilinear interpolation, then center
crop 224×224. The model accepts float32 NHWC `[1,224,224,3]`, values in [0,255]. It
returns float32 `[1,13]` softmax scores in the ordered labels above. Do not apply
ImageNet mean/std normalization again. The label file and complete machine-readable
contract are `ml/mobile/mobilenet_v3_labels.txt` and
`ml/mobile/mobilenet_v3_tflite_contract_v2.json`.

### Training and validation selection

TensorFlow 2.20.0/Keras ran on CPU in this environment. Seed 42, batch size 32,
weighted sparse categorical cross-entropy, and Adam were used. The initial five
epochs used learning rate 1e-4 and selected epoch 5 at 77.65% validation accuracy.
A lower-rate continuation initialized from that validation-selected checkpoint used
five further epochs at 1e-5; validation accuracy improved each epoch to a best of
84.12% (loss 0.49656) at continuation epoch 5. The continuation was chosen over
the first candidate using validation data only. The best continuation checkpoint
is `ml/artifacts/mobilenet_v3_small_phase7_v2_best.keras`; both stage
histories and checkpoints are local ignored artifacts. The checkpoint used for
export was selected before any MobileNet test evaluation.

Commands for a new run use new artifact paths so existing model artifacts are not
overwritten:

~~~bash
python -m ml.src.train_mobilenet_v3 --epochs 5 --batch-size 32 --seed 42 --checkpoint ml/artifacts/mobilenet_v3_new_best.keras --history ml/artifacts/mobilenet_v3_new_training.json
~~~

The continuation command adds `--initial-checkpoint PATH` and uses a different
`--checkpoint` and `--history` output path. Training refuses to overwrite existing
checkpoint/history files.

### Separate MobileNetV3 held-out test result

After selecting the best continuation checkpoint using validation accuracy, the
Keras model was evaluated on the same independent 2,719-image mixed test split used
for the historical Phase 5 baseline. Keras test loss was 0.33134 and accuracy was
89.41%. Per-class results:

| Class | Correct / total | Accuracy |
| --- | ---: | ---: |
| pizza | 237 / 250 | 94.80% |
| hamburger | 193 / 250 | 77.20% |
| sushi | 213 / 250 | 85.20% |
| steak | 196 / 250 | 78.40% |
| ice_cream | 208 / 250 | 83.20% |
| donuts | 228 / 250 | 91.20% |
| omelette | 220 / 250 | 88.00% |
| french_fries | 243 / 250 | 97.20% |
| chicken_wings | 239 / 250 | 95.60% |
| spaghetti_bolognese | 239 / 250 | 95.60% |
| borek | 77 / 78 | 98.72% |
| kebap | 81 / 82 | 98.78% |
| sarma | 57 / 59 | 96.61% |

This is MobileNetV3 classification accuracy, not nutrition/calorie accuracy. The
historical PyTorch EfficientNet-B0 result remains 95.66% on the same mixed test
set; it is not attributed to MobileNetV3.

### TFLite conversion and verification

`export_mobilenet_v3_tflite.py` converts the selected `.keras` model with TFLite
dynamic-range weight optimization. The real flatbuffer is
`ml/artifacts/mobilenet_v3_small_phase7_v2.tflite` (1,113,776 bytes). TFLite has
float32 input and output tensors: input `[1,224,224,3]`; output `[1,13]`.

The serialized artifact loaded in `tf.lite.Interpreter`, allocated tensors, and ran
inference. On all 2,719 held-out images, exported TFLite accuracy was 88.12%; its
per-class results were 93.20% pizza, 74.00% hamburger, 81.20% sushi, 72.40% steak,
83.60% ice cream, 92.80% donuts, 88.40% omelette, 97.20% french fries, 94.80%
chicken wings, 96.40% spaghetti Bolognese, 97.44% borek, 98.78% kebap, and 91.53%
sarma. TFLite was lower than the selected Keras checkpoint by 1.29 percentage
points on this split.
The aggregate verification details are saved in the ignored artifact
`ml/artifacts/mobilenet_v3_small_phase7_v2_tflite_verification.json`.

For 39 representative images (three from each class), TFLite and Keras top-1
predictions agreed for 38/39 images. The mean absolute probability difference was
0.01010; the maximum observed difference was 0.32680. The exporter chose this
quantization because aggregate TFLite test accuracy remained close to the float
checkpoint while the flatbuffer stayed at about 1.1 MB. Confidence remains an
uncalibrated classifier score and never changes nutrition or portion size.

### TFLite confidence-threshold evaluation

`ml/src/evaluate_tflite_confidence.py` evaluates confidence using the deployed
`mobilenet_v3_small_phase7_v2.tflite` artifact and the exact Phase 5 deterministic
evaluation transforms/class order above. It records each image's true class,
top-1 predicted class, maximum softmax confidence, and correctness. Run it first on
validation; choose a threshold there, then run the held-out test split once with
that frozen value:

~~~bash
python -m ml.src.evaluate_tflite_confidence --split validation
python -m ml.src.evaluate_tflite_confidence --split test --threshold 0.55
~~~

The JSON reports, including per-image rows, are written under ignored
`ml/artifacts/`. Threshold comparison is inclusive: confidence greater than or
equal to the threshold is accepted. The validation run considered candidates from
0.30 through 0.70 in 0.05 steps. The recommended candidate is **0.55**: it raises
accepted-prediction accuracy to 89.54% while retaining 85.47% of all predictions;
it rejects 47.95% of originally incorrect predictions and 7.60% of originally
correct predictions. It also rejects a 0.23-confidence result such as the observed
computer-to-donuts example. Compared with 0.50, 0.55 rejects 26 more incorrect
predictions and 28 more correct predictions while increasing accepted accuracy by
1.35 percentage points. Moving to 0.60 rejects only 24 more incorrect predictions
but loses 29 additional correct predictions and 3.12 percentage points of coverage.
That makes 0.55 a practical middle point for this application's trust/coverage
tradeoff. This is a validation-selected operating point, not a claim that confidence
is calibrated.

Validation candidate results (n=1,700; overall accuracy 82.82%):

| Threshold | Accepted | Rejected | Accepted correct | Accepted incorrect | Accepted accuracy | Originally correct rejected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 1,678 (98.71%) | 22 (1.29%) | 1,406 | 272 | 83.79% | 2 (0.14%) |
| 0.35 | 1,648 (96.94%) | 52 (3.06%) | 1,399 | 249 | 84.89% | 9 (0.64%) |
| 0.40 | 1,610 (94.71%) | 90 (5.29%) | 1,382 | 228 | 85.84% | 26 (1.85%) |
| 0.45 | 1,564 (92.00%) | 136 (8.00%) | 1,360 | 204 | 86.96% | 48 (3.41%) |
| 0.50 | 1,507 (88.65%) | 193 (11.35%) | 1,329 | 178 | 88.19% | 79 (5.61%) |
| **0.55** | **1,453 (85.47%)** | **247 (14.53%)** | **1,301** | **152** | **89.54%** | **107 (7.60%)** |
| 0.60 | 1,400 (82.35%) | 300 (17.65%) | 1,272 | 128 | 90.86% | 136 (9.66%) |
| 0.65 | 1,348 (79.29%) | 352 (20.71%) | 1,243 | 105 | 92.21% | 165 (11.72%) |
| 0.70 | 1,290 (75.88%) | 410 (24.12%) | 1,210 | 80 | 93.80% | 198 (14.06%) |

Confidence on validation correct predictions (n=1,408): mean 0.8904, median
0.9751; 10th/25th/75th/90th percentiles 0.6063/0.8483/0.9970/0.9995; range
0.2418–1.0000. On incorrect predictions (n=292): mean 0.5819, median 0.5616;
10th/25th/75th/90th percentiles 0.3200/0.4292/0.7208/0.8979; range
0.1691–0.9992. The overlap is substantial, so any cutoff rejects some correct
food predictions and accepts some mistakes.

With 0.55 frozen before test evaluation, the 2,719-image held-out test set had
2,396 correct and 323 incorrect predictions before rejection (88.12% accuracy).
The threshold accepted 2,410 (88.64%) and rejected 309 (11.36%). Accepted
predictions included 2,240 correct and 170 incorrect (92.95% accepted accuracy).
It rejected 156 of 2,396 originally correct predictions (6.51%) and 153 of 323
incorrect predictions (47.37%). Test results were not used to change the cutoff.
Correct test predictions had mean/median confidence 0.9041/0.9832 (10th/25th/75th/
90th percentiles 0.6371/0.8938/0.9979/0.9997; range 0.2037–1.0000); incorrect
predictions had 0.5887/0.5703 (0.3431/0.4189/0.7335/0.8759; range
0.1977–0.9873).

This cutoff is only a low-confidence rejection rule for a **closed-set** 13-class
classifier. The model always chooses among its supported classes; it is not a
food-versus-non-food detector. Unsupported foods and non-food images can still
receive high confidence and pass the cutoff. Physical testing observed a laptop
classified as Donuts at about 64% confidence and a cake classified as Donuts above
50%; both illustrate why the cutoff is not an unknown-food detector. Flutter uses
the validation-derived 0.55 threshold to reject only low-confidence predictions.

The Python TFLite pipeline is `mobilenet_v3_pipeline.py` and returns canonical
`food_id`, confidence, default serving, and exact catalog nutrients. Its confidence
is informational only. The classifier has no unknown-food output. The Flutter app
in `mobile/` bundles this model and applies the documented 0.55 low-confidence
gate during Scan. This gate does not identify whether an image contains food; only
the 13 supported classes are recognized, and unsupported foods or non-food objects
may still receive high-confidence predictions.

~~~bash
python -m ml.src.evaluate_mobilenet_v3
python -m ml.src.export_mobilenet_v3_tflite
python -m ml.src.verify_mobilenet_v3
python -m ml.src.mobilenet_v3_pipeline path/to/meal.jpg
~~~
