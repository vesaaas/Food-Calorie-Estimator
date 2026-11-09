# ML Pipeline

## Current mobile model

The current Flutter app uses a float32 TensorFlow Lite MobileNetV3 Small classifier
with 14 outputs. Indices 0–12 are the 13 supported food classes; index 13 is
`non_food`. The exact ordered labels are:

```text
pizza, hamburger, sushi, steak, ice_cream, donuts, omelette,
french_fries, chicken_wings, spaghetti_bolognese, borek, kebap, sarma, non_food
```

The Flutter asset is
`mobile/assets/models/mobilenet_v3_small_phase9_5_14class.tflite`; labels are in
`mobile/assets/models/labels_phase9_5_14class.txt`. The previous 13-class model,
`mobile/assets/models/mobilenet_v3_small_phase7_v2.tflite`, is retained for rollback.
The 14-class model is the current app model. Android v1.1.0 (`1.1.0+2`) is
published; v1.1.1 (`1.1.1+3`) is a maintenance candidate with no model or prediction
changes. The published v1.1.0 APK passed physical Samsung device testing. See
[`ANDROID_RELEASE.md`](ANDROID_RELEASE.md).

## Preprocessing and output contract

Images are oriented using EXIF metadata, decoded to RGB, resized so the shortest
edge is 256 pixels with bilinear interpolation, then center-cropped to 224×224.
The model input is float32 NHWC `[1,224,224,3]` in the `[0,255]` pixel range. The
TFLite graph retains MobileNetV3's internal rescaling to `[-1,1]`; callers must not
apply additional normalization. Output is 14 float32 softmax probabilities in the
label order above.

The frozen decision policy is argmax over all outputs. There is no additional
confidence threshold: the former 0.55 acceptance gate is not used by the current
14-class model. Confidence is informational only. If argmax is `non_food`, the app
rejects the image before nutrition lookup or diary saving. For a supported class,
the app requires user confirmation before lookup; a user may retry or manually
select and confirm a supported food. A user can also decline with “My food isn't
listed.” This flow does not convert unsupported foods into `non_food` and is not a
reliable unsupported-food detector.

## Phase 9.3–9.5 experiment and held-out evaluation

The 14-class candidate was prepared from the existing 13 supported-food classes and
a broad `non_food` class. Unsupported foods were retained as a separate evaluation
category, not relabeled as `non_food`. The selected 14-class Keras checkpoint was
evaluated on held-out cohorts of 2,719 supported-food images, 529 non-food images,
and 22,750 unsupported-food images. The decision policy was frozen as argmax before
this one-time held-out evaluation; held-out results were not used for threshold
selection or model tuning.

Held-out results for the selected candidate:

| Metric | Result |
| --- | ---: |
| Supported-food classification accuracy | **90.33%** |
| Supported-food acceptance | **99.63%** |
| Non-food rejection | **94.33%** |
| Unsupported-food rejection | **2.03%** |

Supported-food acceptance is the share of supported examples assigned any of the
13 supported labels. It is not the same as top-1 classification accuracy: accepted
images may still have the wrong food label. Non-food rejection is the share of
non-food objects assigned to `non_food`. Unsupported-food rejection is measured
separately and remains very low; the model is **not a reliable unsupported-food
detector**. Unsupported foods can receive high-confidence predictions in one of the
supported classes.

Project engineering targets (not reviewer-specified requirements) included at least
97% supported-food acceptance, 95% acceptance per supported class, 80% non-food
rejection, 50% unsupported-food rejection, and no more than a 2 percentage-point
classification-accuracy drop from the 13-class Keras baseline. The full target set
was already unmet during validation: minimum per-class acceptance was 94.67% and
unsupported-food rejection was low. Held-out evaluation was performed on the frozen
candidate and policy, without retuning.

### Keras/TFLite parity and baseline distinction

The Phase 9.5 float32 export was made from the selected 14-class Keras checkpoint.
Deterministic validation samples covered supported food, non-food, and unsupported
food; conversion parity checks passed. The frozen held-out evaluation reported the
same aggregate metrics for the exported TFLite candidate and its Keras checkpoint.
This parity result applies to the 14-class Keras/TFLite pair.

For comparison, the existing 13-class Keras baseline scored **89.41%** on the same
2,719 held-out supported-food examples. The previously deployed 13-class TFLite
baseline scored **88.12%** on those supported-food images. Keras and TFLite results
are separate runtimes and are not treated as interchangeable. The 14-class TFLite
supported-food accuracy is **90.33%**.

Preprocessing parity fixtures compare Python/Pillow and Flutter inputs on five
deterministic images. They record exact matches on three, a maximum pixel difference
of 1 on the square case, and a maximum of 3 on the EXIF-rotated JPEG. The five
fixture argmax outputs matched, with maximum probability-vector difference 0.002416;
all five fixture images predicted `non_food`. This checks the preprocessing/runtime
path, not class-wide accuracy. See
[`../mobile/test/fixtures/phase9_6a/README.md`](../mobile/test/fixtures/phase9_6a/README.md).

## Training and experiment artifacts

Phase 9.3 used transfer learning initialized from the selected 13-class Keras
checkpoint, first training a replacement 14-output head with a frozen backbone and
then optionally fine-tuning a bounded final backbone section. Phase 9.4 evaluated
the frozen selected checkpoint once on held-out cohorts. Phase 9.5 exported the
candidate separately; the production 13-class asset was preserved. Experiment
scripts/configuration are under `ml/experiments/`. Generated datasets, checkpoints,
reports, and isolated model outputs are under Git-ignored data/artifact locations.
No Phase 9 experiment replaces the old rollback model.
The dataset sources and configured train/validation/held-out cohort counts are
documented in [`DATASET.md`](DATASET.md).

## Historical 13-class baseline

The former 13-class MobileNetV3 model achieved **88.12% TFLite top-1 accuracy** on
2,719 supported-food test images; its Keras checkpoint scored **89.41%** on the same
cohort. Those are baseline results, not metrics for the 14-class model. Historical
13-class training, data split methodology, and per-class results are preserved in
the experiment artifacts and repository history. The previous 0.55 confidence
policy was evaluated for that closed-set model; it is not part of the current
14-class argmax decision policy.

## Limitations

The 14-class classifier's `non_food` label rejects many non-food objects but does
not identify unsupported foods reliably. Its 2.03% held-out unsupported-food
rejection means most such examples were assigned a supported food label. User
confirmation is a practical safeguard, not an unsupported-food detector. Classifier
confidence does not adjust nutrition. Portion weight is user-entered, not estimated
from pixels. Nutrition is based on representative catalog mappings, not an exact
measurement of recipe composition.
