import '../models/diary_state.dart';

abstract interface class DiaryRepository {
  Future<DiaryState> load();

  Future<void> save(DiaryState state);
}
