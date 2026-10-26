import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

import '../models/diary_state.dart';
import '../models/meal_entry.dart';
import 'diary_repository.dart';
import 'json_diary_repository.dart';

typedef DatabasePathProvider = Future<String> Function();

/// SQLite persistence adapter. The legacy JSON file is kept as a migration
/// source and is never removed by this repository.
class SqliteDiaryRepository implements DiaryRepository {
  SqliteDiaryRepository({
    DatabaseFactory? databaseFactory,
    DatabasePathProvider? databasePathProvider,
    JsonDiaryRepository? legacyRepository,
    this.databaseName = 'food_calorie_diary.sqlite',
  }) : _databaseFactory = databaseFactory ?? databaseFactorySqflitePlugin,
       _databasePathProvider = databasePathProvider ?? _defaultDatabasePath,
       _legacyRepository = legacyRepository ?? JsonDiaryRepository() {
    if (databaseName.isEmpty || databaseName.contains(Platform.pathSeparator)) {
      throw ArgumentError.value(
        databaseName,
        'databaseName',
        'Must be a file name.',
      );
    }
  }

  static const schemaVersion = 1;
  static const _migrationKey = 'legacy_json_migration_v1_complete';

  final DatabaseFactory _databaseFactory;
  final DatabasePathProvider _databasePathProvider;
  final JsonDiaryRepository _legacyRepository;
  final String databaseName;
  Database? _database;
  Future<void>? _migration;

  static Future<String> _defaultDatabasePath() async {
    final directory = await getApplicationSupportDirectory();
    return directory.path;
  }

  @override
  Future<DiaryState> load() async {
    final db = await _open();
    await _migrateLegacyJsonOnce(db);
    final mealRows = await db.query('meals', orderBy: 'sort_order ASC');
    final goalRows = await db.query(
      'goals',
      where: 'goal_id = ?',
      whereArgs: [1],
    );
    return DiaryState(
      meals: mealRows.map(_mealFromRow),
      dailyCalorieGoal: goalRows.isEmpty
          ? null
          : (goalRows.single['daily_calorie_goal'] as num).toDouble(),
    );
  }

  @override
  Future<void> save(DiaryState state) async {
    final db = await _open();
    await _migrateLegacyJsonOnce(db);
    await db.transaction((txn) async {
      await txn.delete('meals');
      for (var index = 0; index < state.meals.length; index++) {
        await txn.insert('meals', _mealToRow(state.meals[index], index));
      }
      await txn.delete('goals', where: 'goal_id = ?', whereArgs: [1]);
      if (state.dailyCalorieGoal != null) {
        await txn.insert('goals', {
          'goal_id': 1,
          'daily_calorie_goal': state.dailyCalorieGoal,
        });
      }
    });
  }

  Future<void> close() async {
    final db = _database;
    _database = null;
    if (db != null) await db.close();
  }

  Future<Database> _open() async {
    final existing = _database;
    if (existing != null) return existing;
    final directory = await _databasePathProvider();
    final separator = Platform.pathSeparator;
    final path = directory.endsWith(separator)
        ? '$directory$databaseName'
        : '$directory$separator$databaseName';
    await Directory(directory).create(recursive: true);
    final db = await _databaseFactory.openDatabase(
      path,
      options: OpenDatabaseOptions(
        version: schemaVersion,
        onCreate: (database, version) async {
          await database.execute('''
            CREATE TABLE meals (
              meal_id TEXT PRIMARY KEY NOT NULL,
              sort_order INTEGER NOT NULL,
              food_id TEXT NOT NULL,
              display_name TEXT NOT NULL,
              logged_at_utc TEXT NOT NULL,
              portion_grams REAL NOT NULL,
              portion_mode TEXT NOT NULL,
              calories_kcal REAL NOT NULL,
              protein_g REAL NOT NULL,
              carbohydrates_g REAL NOT NULL,
              fat_g REAL NOT NULL,
              classifier_confidence REAL,
              photo_reference TEXT
            )
          ''');
          await database.execute(
            'CREATE INDEX idx_meals_logged_at_utc ON meals(logged_at_utc)',
          );
          await database.execute('''
            CREATE TABLE goals (
              goal_id INTEGER PRIMARY KEY CHECK (goal_id = 1),
              daily_calorie_goal REAL NOT NULL
            )
          ''');
          await database.execute('''
            CREATE TABLE app_metadata (
              metadata_key TEXT PRIMARY KEY NOT NULL,
              metadata_value TEXT NOT NULL
            )
          ''');
        },
        onUpgrade: (database, oldVersion, newVersion) async {
          // Add forward-only schema upgrades here; existing records are retained.
        },
      ),
    );
    _database = db;
    return db;
  }

  Future<void> _migrateLegacyJsonOnce(Database db) async {
    final pending = _migration;
    if (pending != null) return pending;
    final completer = _runMigration(db);
    _migration = completer;
    try {
      await completer;
    } catch (_) {
      _migration = null;
      rethrow;
    }
  }

  Future<void> _runMigration(Database db) async {
    final completed = await db.query(
      'app_metadata',
      columns: ['metadata_value'],
      where: 'metadata_key = ?',
      whereArgs: [_migrationKey],
      limit: 1,
    );
    if (completed.isNotEmpty) return;

    // Parse and validate the source before opening the write transaction. Any
    // failure leaves the original JSON intact and the migration marker unset.
    final legacy = await _legacyRepository.load();
    await db.transaction((txn) async {
      final alreadyCompleted = await txn.query(
        'app_metadata',
        where: 'metadata_key = ?',
        whereArgs: [_migrationKey],
        limit: 1,
      );
      if (alreadyCompleted.isNotEmpty) return;
      for (var index = 0; index < legacy.meals.length; index++) {
        await txn.insert('meals', _mealToRow(legacy.meals[index], index));
      }
      if (legacy.dailyCalorieGoal != null) {
        await txn.insert('goals', {
          'goal_id': 1,
          'daily_calorie_goal': legacy.dailyCalorieGoal,
        });
      }
      await txn.insert('app_metadata', {
        'metadata_key': _migrationKey,
        'metadata_value': '1',
      });
    });
  }

  static Map<String, Object?> _mealToRow(MealEntry meal, int sortOrder) => {
    'meal_id': meal.mealId,
    'sort_order': sortOrder,
    'food_id': meal.foodId,
    'display_name': meal.displayName,
    'logged_at_utc': meal.loggedAt.toUtc().toIso8601String(),
    'portion_grams': meal.portionGrams,
    'portion_mode': meal.portionMode,
    'calories_kcal': meal.caloriesKcal,
    'protein_g': meal.proteinGrams,
    'carbohydrates_g': meal.carbohydratesGrams,
    'fat_g': meal.fatGrams,
    'classifier_confidence': meal.classifierConfidence,
    'photo_reference': meal.photoReference,
  };

  static MealEntry _mealFromRow(Map<String, Object?> row) => MealEntry(
    mealId: row['meal_id']! as String,
    foodId: row['food_id']! as String,
    displayName: row['display_name']! as String,
    loggedAt: DateTime.parse(row['logged_at_utc']! as String),
    portionGrams: (row['portion_grams']! as num).toDouble(),
    portionMode: row['portion_mode']! as String,
    caloriesKcal: (row['calories_kcal']! as num).toDouble(),
    proteinGrams: (row['protein_g']! as num).toDouble(),
    carbohydratesGrams: (row['carbohydrates_g']! as num).toDouble(),
    fatGrams: (row['fat_g']! as num).toDouble(),
    classifierConfidence: (row['classifier_confidence'] as num?)?.toDouble(),
    photoReference: row['photo_reference'] as String?,
  );
}
