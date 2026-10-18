# Architecture

## Current and planned flow

~~~text
Food image
    ↓
Image preprocessing
    ↓
Food classification model       ← Phase 3
    ↓
Predicted Food-101 class
    ↓
Nutrition/calorie data layer    ← Phase 4
    ↓
Application/API/mobile layer    ← Phases 5–6
~~~

## Responsibilities

### Image input

A future application or API will provide a meal image. Phase 2 only handles images
already represented by Food-101 and does not implement upload or camera behavior.

### Image preprocessing

ml/src/preprocessing.py converts images to RGB tensors, applies training or
evaluation transforms, resizes them to model-ready dimensions, and applies ImageNet
normalization. ml/src/dataset.py attaches the appropriate transform to each split.

### Food classification model

Phase 3 will consume the DataLoaders produced by ml/src/dataset.py and predict one
of the ten development classes. The current repository contains no model or training
loop.

### Nutrition/calorie data layer

Phase 4 will map a predicted food class to nutrition information using a separate
data source and explicit assumptions about serving size. Food-101 provides food
classification labels only; it does not directly provide calories, portion weights,
macronutrients, or nutrition values.

### Application, API, and mobile layers

Later phases may expose inference through an API and connect it to the mobile
application. Those components should consume a model prediction and nutrition
lookup result through stable interfaces. They are currently placeholders represented
by the existing mobile/ and nutrition/ directories.

## Phase 2 boundary

Phase 2 owns dataset download/reuse, metadata validation, class filtering, split
construction, transforms, DataLoaders, exploration, and verification. It does not
choose a model architecture, train weights, estimate calories, or implement a user
interface.
