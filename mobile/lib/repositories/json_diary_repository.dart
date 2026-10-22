import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:path_provider/path_provider.dart';

import '../models/diary_state.dart';
import 'diary_repository.dart';

typedef DiaryDirectoryProvider = Future<Directory> Function();

class DiaryPersistenceException implements Exception {
  const DiaryPersistenceException(this.message);

  final String message;

  @override
  String toString() => message;
}

class JsonDiaryRepository implements DiaryRepository {
  JsonDiaryRepository({
    DiaryDirectoryProvider? directoryProvider,
    this.fileName = 'food_calorie_diary.json',
  }) : _directoryProvider =
           directoryProvider ?? getApplicationSupportDirectory {
    if (fileName.isEmpty || fileName.contains(Platform.pathSeparator)) {
      throw ArgumentError.value(fileName, 'fileName', 'Must be a file name.');
    }
  }

  final DiaryDirectoryProvider _directoryProvider;
  final String fileName;
  final Random _random = Random.secure();

  @override
  Future<DiaryState> load() async {
    try {
      final file = await _diaryFile();
      if (!await file.exists()) return DiaryState.empty;
      final source = await file.readAsString();
      _JsonKeyScanner(source).check();
      final decoded = jsonDecode(source);
      if (decoded is! Map<String, dynamic>) {
        throw const FormatException('Diary state must be a JSON object.');
      }
      return DiaryState.fromJson(decoded);
    } on DiaryPersistenceException {
      rethrow;
    } on Object catch (error) {
      throw DiaryPersistenceException(
        'Could not safely load the saved diary: $error',
      );
    }
  }

  @override
  Future<void> save(DiaryState state) async {
    File? temporary;
    try {
      final target = await _diaryFile();
      await target.parent.create(recursive: true);
      final suffix = _random.nextInt(1 << 32).toRadixString(16);
      temporary = File(
        '${target.parent.path}${Platform.pathSeparator}.$fileName.$suffix.tmp',
      );
      await temporary.writeAsString(
        '${jsonEncode(state.toJson())}\n',
        flush: true,
      );
      // The temporary file shares the target directory, so supported platforms
      // replace by a same-volume rename after the complete write is flushed.
      await temporary.rename(target.path);
      temporary = null;
    } on Object catch (error) {
      throw DiaryPersistenceException(
        'Could not save the diary. Please try again: $error',
      );
    } finally {
      if (temporary != null) {
        try {
          if (await temporary.exists()) await temporary.delete();
        } on Object {
          // Preserve the original read/write failure if cleanup also fails.
        }
      }
    }
  }

  Future<File> _diaryFile() async {
    final directory = await _directoryProvider();
    return File('${directory.path}${Platform.pathSeparator}$fileName');
  }
}

/// Checks object keys before dart:convert decodes maps and collapses duplicates.
class _JsonKeyScanner {
  _JsonKeyScanner(this.source);

  final String source;
  var _offset = 0;

  void check() {
    _value();
    _space();
    if (_offset != source.length) {
      throw const FormatException('Unexpected data after the diary JSON.');
    }
  }

  void _value() {
    _space();
    if (_offset >= source.length) {
      throw const FormatException('Unexpected end of diary JSON.');
    }
    switch (source[_offset]) {
      case '{':
        _object();
      case '[':
        _array();
      case '"':
        _string();
      default:
        _primitive();
    }
  }

  void _object() {
    _offset++;
    _space();
    if (_consume('}')) return;
    final keys = <String>{};
    while (true) {
      _space();
      if (_offset >= source.length || source[_offset] != '"') {
        throw const FormatException('Invalid object key in diary JSON.');
      }
      final key = _string();
      if (!keys.add(key)) {
        throw FormatException('Duplicate JSON object key: $key');
      }
      _space();
      if (!_consume(':')) {
        throw const FormatException('Missing colon in diary JSON object.');
      }
      _value();
      _space();
      if (_consume('}')) return;
      if (!_consume(',')) {
        throw const FormatException('Invalid diary JSON object separator.');
      }
    }
  }

  void _array() {
    _offset++;
    _space();
    if (_consume(']')) return;
    while (true) {
      _value();
      _space();
      if (_consume(']')) return;
      if (!_consume(',')) {
        throw const FormatException('Invalid diary JSON array separator.');
      }
    }
  }

  String _string() {
    final start = _offset;
    _offset++;
    while (_offset < source.length) {
      final character = source[_offset++];
      if (character == '\\') {
        if (_offset >= source.length) break;
        _offset++;
      } else if (character == '"') {
        final decoded = jsonDecode(source.substring(start, _offset));
        if (decoded is! String) {
          throw const FormatException('Invalid string in diary JSON.');
        }
        return decoded;
      }
    }
    throw const FormatException('Unterminated string in diary JSON.');
  }

  void _primitive() {
    final start = _offset;
    while (_offset < source.length && !',]} \t\r\n'.contains(source[_offset])) {
      _offset++;
    }
    if (_offset == start) {
      throw const FormatException('Invalid value in diary JSON.');
    }
  }

  void _space() {
    while (_offset < source.length && ' \t\r\n'.contains(source[_offset])) {
      _offset++;
    }
  }

  bool _consume(String character) {
    if (_offset < source.length && source[_offset] == character) {
      _offset++;
      return true;
    }
    return false;
  }
}
