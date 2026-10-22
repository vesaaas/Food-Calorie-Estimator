import 'package:flutter/material.dart';

import '../services/diary_controller.dart';
import '../widgets/screen_header.dart';

class GoalScreen extends StatefulWidget {
  const GoalScreen({super.key, required this.diaryController});

  final DiaryController diaryController;

  @override
  State<GoalScreen> createState() => _GoalScreenState();
}

class _GoalScreenState extends State<GoalScreen> {
  final _formKey = GlobalKey<FormState>();
  final _controller = TextEditingController();
  bool _isSaving = false;
  bool _goalFieldLoaded = false;

  @override
  void initState() {
    super.initState();
    widget.diaryController.addListener(_loadSavedGoal);
    _loadSavedGoal();
  }

  @override
  void dispose() {
    widget.diaryController.removeListener(_loadSavedGoal);
    _controller.dispose();
    super.dispose();
  }

  void _loadSavedGoal() {
    if (_goalFieldLoaded || !widget.diaryController.isInitialized) return;
    final goal = widget.diaryController.dailyCalorieGoal;
    _controller.text = goal == null ? '' : _formatGoal(goal);
    _goalFieldLoaded = true;
  }

  Future<void> _saveGoal() async {
    if (!_formKey.currentState!.validate()) return;
    final goal = double.parse(_controller.text.trim());
    setState(() => _isSaving = true);
    try {
      await widget.diaryController.setDailyCalorieGoal(goal);
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(const SnackBar(content: Text('Daily goal saved.')));
    } on Object {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          const SnackBar(
            content: Text('Could not save the goal. Please try again.'),
          ),
        );
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.diaryController,
      builder: (context, _) => ListView(
        keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
        padding: const EdgeInsets.fromLTRB(22, 24, 22, 28),
        children: [
          const ScreenHeader(
            title: 'Daily goal',
            subtitle: 'Set a personal calorie target.',
          ),
          const SizedBox(height: 24),
          if (!widget.diaryController.isInitialized)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    Text(
                      widget.diaryController.isLoading
                          ? 'Loading your saved goal…'
                          : 'Your saved goal could not be loaded.',
                    ),
                    if (!widget.diaryController.isLoading) ...[
                      const SizedBox(height: 12),
                      OutlinedButton(
                        onPressed: widget.diaryController.initialize,
                        child: const Text('Try again'),
                      ),
                    ],
                  ],
                ),
              ),
            )
          else
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Calories per day',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Choose your own daily calorie target. It does not change food nutrition estimates.',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 18),
                      TextFormField(
                        key: const Key('daily-goal-field'),
                        controller: _controller,
                        onTapOutside: (_) =>
                            FocusManager.instance.primaryFocus?.unfocus(),
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                        ),
                        textInputAction: TextInputAction.done,
                        decoration: const InputDecoration(
                          labelText: 'Daily calorie goal',
                          suffixText: 'kcal',
                        ),
                        validator: (value) {
                          final parsed = double.tryParse(value?.trim() ?? '');
                          if (parsed == null ||
                              !parsed.isFinite ||
                              parsed <= 0) {
                            return 'Enter a number greater than zero';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: 16),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          key: const Key('save-daily-goal'),
                          onPressed: _isSaving ? null : _saveGoal,
                          child: Text(_isSaving ? 'Saving…' : 'Save goal'),
                        ),
                      ),
                      const SizedBox(height: 12),
                      if (widget.diaryController.dailyCalorieGoal
                          case final goal?)
                        Text('Current goal: ${_formatGoal(goal)} kcal'),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

String _formatGoal(double value) =>
    value == value.roundToDouble() ? value.toStringAsFixed(0) : '$value';
