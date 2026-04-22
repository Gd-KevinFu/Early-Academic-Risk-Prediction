# Student Label Generation Pipeline

This repository provides a single-file, directly runnable pipeline for generating nine student-level labels from learning behavior, campus card records, exam data, and a student linkage table.

## What this project does

The pipeline integrates data from four sources:

- **OULAD** online learning interaction logs
- **CCSD** campus card transaction records
- **exam records** with course-credit information
- **student linkage data** that aligns IDs across systems

It outputs nine labels for each student and supports reproducible machine-learning workflows through:

- fixed train / validation / test splits at the student level
- train-only fitting of distribution-dependent statistics
- frozen thresholds that can be reused on validation, test, or future cohorts

## Release naming

This checked repository version uses the **GradeU37** output tag. The default exported file names therefore use `GradeU37`, for example `student_nine_labels_train_GradeU37.csv`.

## Nine labels

| ID  | English column key                            | Chinese label name | English label name                          |
| --- | --------------------------------------------- | ------------------ | ------------------------------------------- |
| 1   | `online_learning_engagement_trajectory_type`  | 在线学习参与轨迹类型         | Online Learning Engagement Trajectory Type  |
| 2   | `learning_time_investment_level`              | 学习时间投入水平           | Learning Time Investment Level              |
| 3   | `academic_procrastination_tendency`           | 学业拖延倾向             | Academic Procrastination Tendency           |
| 4   | `online_learning_persistence_level`           | 在线坚持度              | Online Learning Persistence Level           |
| 5   | `daily_routine_stability_level`               | 生活作息稳定性            | Daily Routine Stability Level               |
| 6   | `late_time_consumption_tendency`              | 偏晚消费倾向             | Late-Time Consumption Tendency              |
| 7   | `health_related_behavior_and_lifestyle_index` | 健康相关行为与生活方式指数      | Health-Related Behavior and Lifestyle Index |
| 8   | `academic_life_composite_risk_signal_index`   | 学业–生活综合风险信号指数      | Academic-Life Composite Risk Signal Index   |
| 9   | `academic_lifestyle_synergy_type`             | 学业–生活方式协同类型        | Academic-Lifestyle Synergy Type             |

## Repository files

- `bulid 9 labels_frozen_split_teachingweeks.py`: main checked pipeline script
- `requirements.txt`: minimal Python dependencies
- `label_column_mapping.csv`: English / Chinese / legacy column mapping
- `label_column_mapping.json`: structured label metadata
- `CHECK_REPORT.md`: static validation and consistency report for this release

## Expected folder structure

```text
.
├── OULAD/
│   ├── studentVle.csv
│   ├── studentInfo.csv
│   ├── assessments.csv
│   ├── studentAssessment.csv
│   ├── courses.csv
│   └── studentRegistration.csv
├── CCSD/
│   ├── Data_Campus_card.txt
│   └── Data_Exam.txt
├── dataset/
│   ├── Student_link_generated2.csv
│   └── link table.csv
└── bulid 9 labels_frozen_split_teachingweeks.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. First run: create splits and fit frozen statistics

```bash
python bulid 9 labels_frozen_split_teachingweeks.py \
  --split_dir splits \
  --seed 35
```

### 2. Reuse existing splits and frozen statistics

```bash
python bulid 9 labels_frozen_split_teachingweeks.py \
  --split_dir splits \
  --apply_only \
  --frozen_stats frozen_thresholds.json
```

### 3. Apply frozen statistics to a new cohort without splitting

```bash
python bulid 9 labels_frozen_split_teachingweeks.py \
  --apply_only \
  --skip_split \
  --frozen_stats frozen_thresholds.json
```

## Outputs

The pipeline exports:

- `student_nine_labels_GradeU37.csv`
- `student_nine_labels_train_GradeU37.csv`
- `student_nine_labels_val_GradeU37.csv`
- `student_nine_labels_test_GradeU37.csv`
- `frozen_thresholds.json`
- `label_column_mapping.csv`
- `label_column_mapping.json`
- several auxiliary CSV files used for inspection

The main label table uses stable **English column keys** for open-source compatibility. The Chinese label names are preserved in the mapping files.

## Design notes

### Fixed student-level splits

All splits are generated at the `sid` level. Once saved, the same split files can be reused to keep experiments comparable across runs.

### Train-only frozen statistics

Any step that depends on a sample distribution is fitted on the training split only, for example:

- z-score mean and standard deviation
- quartile thresholds used for discretization
- median thresholds used in binary type assignment

This avoids leakage from validation and test data.

### Column naming policy

The code now uses:

- stable English keys in exported data tables
- exact Chinese label names in metadata and documentation
- legacy keys preserved in mapping files for backward reference

## Notes for open-source release

- The script keeps the original data path assumptions to preserve runtime compatibility.
- Some source columns remain in Chinese because they are part of the raw datasets.
- The body of the script has been reduced to a single effective implementation, with legacy duplicate code removed.
- The documentation and label names were checked against the appendix and the original source file before this release package was generated.
