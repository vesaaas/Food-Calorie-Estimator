import 'dart:io';
import 'dart:math';

import 'package:path_provider/path_provider.dart';

typedef MealPhotoDirectoryProvider = Future<Directory> Function();

abstract interface class MealPhotoStore {
  Future<String> savePhoto(String sourcePath);

  Future<String?> resolvePhotoPath(String? photoReference);

  Future<void> deletePhoto(String? photoReference);

  Future<void> removeUnreferencedPhotos(Set<String> referencedPhotos);
}

/// Stores photos under Application Support and persists only opaque file names.
class AppSupportMealPhotoStore implements MealPhotoStore {
  AppSupportMealPhotoStore({
    MealPhotoDirectoryProvider? directoryProvider,
    Random? random,
  }) : _directoryProvider = directoryProvider ?? getApplicationSupportDirectory,
       _random = random ?? Random.secure();

  final MealPhotoDirectoryProvider _directoryProvider;
  final Random _random;

  @override
  Future<String> savePhoto(String sourcePath) async {
    final extension = _supportedExtension(sourcePath);
    final source = File(sourcePath);
    if (!await source.exists() || await source.length() == 0) {
      throw const FileSystemException('Selected meal photo is unavailable.');
    }
    final directory = await _photoDirectory();
    String? reference;
    File? temporary;
    try {
      for (var attempt = 0; attempt < 5; attempt++) {
        final token = List<int>.generate(
          16,
          (_) => _random.nextInt(256),
        ).map((byte) => byte.toRadixString(16).padLeft(2, '0')).join();
        final candidate = '$token.$extension';
        if (!await File(_join(directory.path, candidate)).exists()) {
          reference = candidate;
          break;
        }
      }
      if (reference == null) {
        throw const FileSystemException(
          'Could not allocate a meal photo name.',
        );
      }
      temporary = File(_join(directory.path, '.$reference.tmp'));
      await source.copy(temporary.path);
      if (await temporary.length() == 0) {
        throw const FileSystemException('Copied meal photo is empty.');
      }
      final stored = await temporary.rename(_join(directory.path, reference));
      temporary = null;
      return stored.uri.pathSegments.last;
    } finally {
      if (temporary != null && await temporary.exists()) {
        await temporary.delete();
      }
    }
  }

  @override
  Future<String?> resolvePhotoPath(String? photoReference) async {
    if (!_isSafePhotoReference(photoReference)) return null;
    final directory = await _photoDirectory();
    final file = File(_join(directory.path, photoReference!));
    if (await FileSystemEntity.type(file.path, followLinks: false) !=
        FileSystemEntityType.file) {
      return null;
    }
    return file.path;
  }

  @override
  Future<void> deletePhoto(String? photoReference) async {
    if (!_isSafePhotoReference(photoReference)) return;
    final directory = await _photoDirectory();
    final file = File(_join(directory.path, photoReference!));
    if (await FileSystemEntity.type(file.path, followLinks: false) ==
        FileSystemEntityType.file) {
      await file.delete();
    }
  }

  @override
  Future<void> removeUnreferencedPhotos(Set<String> referencedPhotos) async {
    final directory = await _photoDirectory();
    await for (final entity in directory.list(followLinks: false)) {
      if (await FileSystemEntity.type(entity.path, followLinks: false) !=
          FileSystemEntityType.file) {
        continue;
      }
      final name = entity.uri.pathSegments.last;
      final temporaryMatch = name.startsWith('.') && name.endsWith('.tmp')
          ? name.substring(1, name.length - 4)
          : null;
      final reference = temporaryMatch ?? name;
      if (!_isSafePhotoReference(reference)) continue;
      if (temporaryMatch != null || !referencedPhotos.contains(reference)) {
        await File(entity.path).delete();
      }
    }
  }

  Future<Directory> _photoDirectory() async {
    final support = await _directoryProvider();
    final directory = Directory(_join(support.path, 'meal_photos'));
    await directory.create(recursive: true);
    if (await FileSystemEntity.type(directory.path, followLinks: false) !=
        FileSystemEntityType.directory) {
      throw const FileSystemException('Meal photo storage is not a directory.');
    }
    return directory;
  }
}

/// Test/default controller adapter for callers that never attach photo files.
class NoopMealPhotoStore implements MealPhotoStore {
  const NoopMealPhotoStore();

  @override
  Future<String> savePhoto(String sourcePath) =>
      Future.error(UnsupportedError('A photo store was not configured.'));

  @override
  Future<String?> resolvePhotoPath(String? photoReference) async => null;

  @override
  Future<void> deletePhoto(String? photoReference) async {}

  @override
  Future<void> removeUnreferencedPhotos(Set<String> referencedPhotos) async {}
}

bool _isSafePhotoReference(String? reference) =>
    reference != null &&
    RegExp(r'^[a-f0-9]{32}\.(jpg|jpeg|png|webp|heic)$').hasMatch(reference);

String _supportedExtension(String path) {
  final name = path.split(Platform.pathSeparator).last;
  final dot = name.lastIndexOf('.');
  final extension = dot < 0 ? '' : name.substring(dot + 1).toLowerCase();
  if (!{'jpg', 'jpeg', 'png', 'webp', 'heic'}.contains(extension)) {
    throw const FileSystemException(
      'Selected meal photo format is unsupported.',
    );
  }
  return extension;
}

String _join(String directory, String name) =>
    '$directory${Platform.pathSeparator}$name';
