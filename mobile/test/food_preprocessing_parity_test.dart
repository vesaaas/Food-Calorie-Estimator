import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/services/food_classification_service.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:image/image.dart' as img;

void main() {
  const cases = <String, List<int>>{
    'landscape': [349, 256, 62, 16],
    'portrait': [256, 349, 16, 62],
    'square': [256, 256, 16, 16],
    'odd_dimensions': [343, 256, 60, 16],
    'exif_rotated': [256, 357, 16, 66],
  };

  for (final entry in cases.entries) {
    test('${entry.key} matches the Pillow preprocessing golden', () async {
      final fixtureDirectory = Directory('test/fixtures/phase9_6a');
      final sourcePath =
          '${fixtureDirectory.path}/${entry.key}.'
          '${entry.key == 'exif_rotated' ? 'jpg' : 'png'}';
      final bytes = await File(sourcePath).readAsBytes();
      final decoded = img.decodeImage(bytes);
      expect(decoded, isNotNull, reason: 'fixture $sourcePath must decode');

      final resized = resizeAndCenterCropFoodImage(decoded!);
      expect([resized.width, resized.height], [224, 224]);
      final expectedGeometry = entry.value;
      final orientation = decoded.exif.imageIfd.orientation ?? 1;
      final orientedWidth = orientation >= 5 && orientation <= 8
          ? decoded.height
          : decoded.width;
      final orientedHeight = orientation >= 5 && orientation <= 8
          ? decoded.width
          : decoded.height;
      final scale =
          256 /
          (orientedWidth < orientedHeight ? orientedWidth : orientedHeight);
      expect((orientedWidth * scale).floor(), expectedGeometry[0]);
      expect((orientedHeight * scale).floor(), expectedGeometry[1]);
      expect(
        _roundPython((expectedGeometry[0] - 224) / 2),
        expectedGeometry[2],
      );
      expect(
        _roundPython((expectedGeometry[1] - 224) / 2),
        expectedGeometry[3],
      );

      final actual = phase95FoodImageToFloatRgb(decoded);
      final golden = await File('${fixtureDirectory.path}/${entry.key}.rgb')
          .readAsBytes();
      expect(actual, hasLength(224 * 224 * 3));
      expect(golden, hasLength(224 * 224 * 3));

      var maxDifference = 0.0;
      var sumDifference = 0.0;
      for (var index = 0; index < golden.length; index++) {
        final difference = (actual[index] - golden[index]).abs();
        if (difference > maxDifference) maxDifference = difference;
        sumDifference += difference;
      }
      final meanDifference = sumDifference / golden.length;
      // Test output logs make the measured cross-library error explicit.
      // ignore: avoid_print
      print(
        '${entry.key}: max_abs_pixel_diff=$maxDifference, '
        'mean_abs_pixel_diff=$meanDifference',
      );
      // Pillow and image package use separate bilinear implementations. These
      // limits detect channel/order, geometry, or large resampling drift while
      // allowing only small integer-rounding differences.
      expect(maxDifference, lessThanOrEqualTo(4), reason: entry.key);
      expect(meanDifference, lessThanOrEqualTo(0.6), reason: entry.key);
      expect(
        actual.every((pixel) => pixel.isFinite && pixel >= 0 && pixel <= 255),
        isTrue,
      );
      // The fixture is RGB, and pixel values must be laid out as interleaved
      // R,G,B values in row-major NHWC order.
      expect(actual.length % 3, 0);
    });
  }

  const modelPath =
      '../ml/artifacts/experiments/phase9_5_14class_tflite/'
      'mobilenet_v3_small_phase9_5_14class.tflite';
  final modelFile = File(modelPath);
  test(
    'isolated 14-class TFLite predictions are stable across pixel differences',
    () async {
      final fixtures = Directory('test/fixtures/phase9_6a');
      final reference = jsonDecode(
        await File('${fixtures.path}/tflite_reference.json').readAsString(),
      ) as Map<String, dynamic>;
      final cases = reference['cases'] as Map<String, dynamic>;
      final interpreter = Interpreter.fromFile(modelFile);
      addTearDown(interpreter.close);
      expect(interpreter.getInputTensor(0).shape, [1, 224, 224, 3]);
      expect(interpreter.getOutputTensor(0).shape, [1, 14]);

      for (final name in cases.keys) {
        final extension = name == 'exif_rotated' ? 'jpg' : 'png';
        final decoded = img.decodeImage(
          await File('${fixtures.path}/$name.$extension').readAsBytes(),
        );
        expect(decoded, isNotNull);
        final pixels = phase95FoodImageToFloatRgb(decoded!);
        final output = Float32List(14);
        interpreter.run(pixels, output);
        final expected = (cases[name] as Map<String, dynamic>);
        final expectedProbabilities = (expected['probabilities'] as List)
            .cast<num>();
        final actualClass = _argmax(output);
        final expectedClass = expected['argmax'] as int;
        var maxProbabilityDifference = 0.0;
        for (var i = 0; i < output.length; i++) {
          final difference = (output[i] - expectedProbabilities[i]).abs();
          if (difference > maxProbabilityDifference) {
            maxProbabilityDifference = difference;
          }
        }
        // ignore: avoid_print
        print(
          '$name: Python/Flutter TFLite argmax '
          '$expectedClass/$actualClass, '
          'max_probability_diff=$maxProbabilityDifference',
        );
        expect(actualClass, expectedClass, reason: name);
        expect(
          maxProbabilityDifference,
          lessThanOrEqualTo(0.005),
          reason: name,
        );
      }
    },
    skip: !modelFile.existsSync()
        ? 'Phase 9.5 isolated TFLite artifact is not present in this checkout.'
        : false,
  );
}

int _roundPython(double value) {
  final lower = value.floor();
  final fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower.isEven ? lower : lower + 1;
}

int _argmax(List<double> values) {
  var best = 0;
  for (var index = 1; index < values.length; index++) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}
