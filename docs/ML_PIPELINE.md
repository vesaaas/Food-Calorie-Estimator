# ML Pipeline

## Production model

The mobile production classifier is **MobileNetV3 Small**, implemented with
TensorFlow/Keras and exported to TensorFlow Lite for on-device inference. It maps
images to 13 supported food classes. The TFLite model achieved **88.12% top-1
accuracy on the held-out set of 2,719 images**. This is food-classification
accuracy, not nutrition or calorie-estimation accuracy.

The tracked model contract is `ml/mobile/mobilenet_v3_tflite_contract_v2.json`; the
ordered labels are in `ml/mobile/mobilenet_v3_labels.txt`. Input is RGB float32 NHWC
`[1,224,224,3]` in `[0,255]`, resized with shortest edge 256 and center-cropped to
224×224. MobileNetV3's built-in preprocessing rescales inputs to `[-1,1]`; callers
must not apply ImageNet mean/std normalization again. Output is `[1,13]` softmax
scores. The exported dynamically weight-optimized flatbuffer is about 1.1 MB.

### Classes

```text
pizza, hamburger, sushi, steak, ice_cream, donuts, omelette,
french_fries, chicken_wings, spaghetti_bolognese, borek, kebap, sarma
```

### Evaluation method

The mixed dataset contains 7,684 training images, 1,700 validation images, and
2,719 held-out test images. Food-101 validation examples are selected only from the
official training split; its official test membership is preserved. Three selected
TurkishFoods-25 classes use the source train/eval/test splits, with
`yaprak_sarma` mapped to `sarma`. Data preparation checks duplicate image content
across splits. Details of dataset provenance and split handling are in
[`DATASET.md`](DATASET.md).

The Keras checkpoint was selected using validation accuracy before the held-out test
was evaluated. On the independent test set, Keras scored 89.41%; the exported TFLite
model scored 88.12%. For 39 representative images (three per class), top-1 outputs
matched on 38; mean absolute probability difference was 0.01010. The test split
contains 2,500 Food-101 images and 219 regional-food images. Per-class TFLite
accuracy was 93.20% pizza, 74.00% hamburger, 81.20% sushi, 72.40% steak, 83.60%
ice cream, 92.80% donuts, 88.40% omelette, 97.20% french fries, 94.80% chicken
wings, 96.40% spaghetti Bolognese, 97.44% börek, 98.78% kebap, and 91.53% sarma.

## Confidence policy and limitations

The app uses **0.55** as a low-confidence handling threshold. This value was selected
on validation data. At that operating point validation coverage was 85.47%, with
89.54% accuracy among accepted predictions. With the threshold frozen, the held-out
test accepted 2,410 of 2,719 predictions (88.64% coverage); accepted accuracy was
92.95%. This policy is a retry prompt for low-confidence results only.

The classifier is closed-set and always assigns one of its 13 classes. Unsupported
foods and non-food images can map to supported classes, including at high confidence.
The 0.55 threshold is **not** an open-set, unsupported-food, or food/non-food
detector. Confidence is not calibrated and does not adjust nutrition. Portion weight
is user-entered and is not estimated from pixels.

## Training and reproducibility

The Keras MobileNetV3 Small model uses ImageNet initialization, a new 13-output
softmax head, and fine-tunes the backbone on the existing mixed training data. The
best checkpoint is selected with validation accuracy. The recorded run used
TensorFlow 2.20/Keras on CPU, seed 42, batch size 32, weighted sparse
categorical cross-entropy, and Adam. It used an initial five-epoch stage at a
learning rate of `1e-4`, then a five-epoch lower-rate continuation at `1e-5`; the
continuation's final epoch was selected at 84.12% validation accuracy. The held-out
test set was not used for checkpoint selection. TensorFlow/Keras training and TFLite
export scripts are under `ml/src/`; generated datasets, checkpoints, and reports
are ignored by Git. The model contract and labels remain tracked so the mobile
inference path is explicit and inspectable.

Useful commands, after preparing datasets and installing `requirements.txt`:

```bash
python -m ml.src.train_mobilenet_v3
python -m ml.src.evaluate_mobilenet_v3
python -m ml.src.export_mobilenet_v3_tflite
python -m ml.src.verify_mobilenet_v3
```

## Historical EfficientNet experiments

Earlier PyTorch EfficientNet-B0 models supported development and evaluation. The
original ten-class Food-101 experiment scored 94.48% on its 2,500-image test split.
A separate 13-class mixed-data experiment scored 95.66% on 2,719 images. These
results describe distinct historical experiments and are **not** the deployed
mobile model's accuracy. They are not directly comparable as a basis for the
production claim. Relevant training, evaluation, and inference code remains in
`ml/src/` for reproducibility.

## Robustness research

Alternative approaches were evaluated but did not meet the criteria for integration.
A MobileNet fine-tuning candidate reached 88.23% test accuracy and improved expected
calibration error, but reduced accuracy in 7 of 13 classes. A three-way supported /
unsupported / non-food triage approach rejected too many valid supported foods. A
binary food/non-food candidate did not meet its supported-food and non-food rejection
criteria. None is part of the app; production remains the validated MobileNetV3
TFLite baseline. The classifier's closed-set limitation therefore remains explicit.
