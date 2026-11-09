# Phase 9.6A preprocessing fixtures

These fixtures were generated from deterministic RGB patterns in Python/Pillow.
The source PNG/JPEG bytes are read by both the Flutter preprocessing tests and
the Python reference pipeline. `*.rgb` files contain the reference 224×224
row-major RGB bytes produced by:

1. `ImageOps.exif_transpose`
2. `convert("RGB")`
3. Resize shortest edge to 256 using `int(size * scale)` dimensions and
   `Image.Resampling.BILINEAR`
4. Center crop 224×224 using Python's ties-to-even `round`

`metadata.json` records source/oriented/resized sizes, crop origins, and golden
SHA-256 values. `tflite_reference.json` stores Python TFLite probabilities on
the Pillow golden tensors for parity checks against Flutter's `flutter_litert`
runtime. The reference outputs were generated with the Phase 9.5 artifact, whose
verified model is also bundled in the Flutter assets.

The fixtures cover landscape, portrait, square, odd-sized, and EXIF orientation
6 JPEG inputs. Their generated patterns make channel order and interpolation
differences visible and reproducible; they are not intended as a classifier
accuracy set. The Phase 9.5 14-class model was subsequently integrated at
`mobile/assets/models/mobilenet_v3_small_phase9_5_14class.tflite`; the previous
13-class asset remains available for rollback.

## Flutter comparison and measured parity

The production path in `mobile/lib/services/food_classification_service.dart`
uses `phase95FoodImageToFloatRgb` before invoking the integrated 14-class TFLite
model. It follows the preprocessing contract measured here. The frozen model
decision policy is argmax over 14 outputs: indices 0–12 are supported foods and
index 13 is `non_food`; there is no confidence acceptance threshold. The old
13-class model is retained only as a rollback asset.

| Contract detail | Phase 9.5 Python reference | New Flutter parity helper |
| --- | --- | --- |
| EXIF orientation | Pillow `ImageOps.exif_transpose` before sizing | Reads EXIF orientation and maps source pixels before sizing |
| Decode / channels | Pillow decode, `convert("RGB")` | `image` package decode, explicit R,G,B extraction |
| Resize geometry | shortest edge 256; positive dimensions truncated with `int` | Same geometry and truncation |
| Interpolation | Pillow bilinear | Pillow-compatible center sampling and separable bilinear weights |
| Crop | 224×224; Python ties-to-even `round` offsets | Same size and ties-to-even offsets |
| Model tensor | float32, RGB, row-major NHWC, [0,255] | Same; model-internal scaling remains responsible for normalization |

Golden comparison results (per channel, 150,528 values per image): landscape,
portrait, and odd-sized PNGs were exact (max/mean difference 0/0); square PNG
max difference 1 and mean 0.0003721; EXIF-rotated JPEG max difference 3 and mean
0.4865274. The remaining JPEG error is consistent with decoder/resampling
rounding differences. On the Phase 9.5 TFLite model, all five fixture
argmax predictions matched Python (5/5); maximum probability-vector
difference was 0.002416. All five happened to predict `non_food`, so this small
synthetic set verifies plumbing and parity behavior, not class-wide accuracy.
