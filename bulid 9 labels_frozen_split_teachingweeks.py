"""Student label generation pipeline for public release.

Overview
--------
This script generates nine student-level labels by integrating: 
1. OULAD online learning interaction logs,
2. CCSD campus card transaction records,
3. exam records, and
4. a cross-system student linkage table.

The pipeline is designed for reproducible research and open-source release:
- train/validation/test splits are created at the student level and can be reused;
- all distribution-dependent statistics are fitted on the training split only;
- fitted statistics are frozen and then applied to validation, test, or future cohorts;
- the file contains a single runnable implementation without legacy duplicate code.

Usage summary
-------------
First run (create splits and fit frozen statistics):
    python bulid6_frozen_split_teachingweeks_newlabels_no_sport_v2_repo.py --split_dir splits --seed 35

Reuse existing splits and frozen statistics:
    python bulid6_frozen_split_teachingweeks_newlabels_no_sport_v2_repo.py --split_dir splits --apply_only --frozen_stats frozen_thresholds.json

Apply frozen statistics to a new cohort without splitting:
    python bulid6_frozen_split_teachingweeks_newlabels_no_sport_v2_repo.py --apply_only --skip_split --frozen_stats frozen_thresholds.json

Main outputs
------------
- split files under `split_dir`
- a full student label table with English column names
- train/validation/test subset files
- `label_column_mapping.csv` and `label_column_mapping.json`
- `frozen_thresholds.json` with train-only fitted statistics

Release note
------------
This public-release version uses the `GradeU37` output tag to match the
37-week configuration used in the current experiments and exported label files.

Nine labels
-----------
1. 在线学习参与轨迹类型
2. 学习时间投入水平
3. 学业拖延倾向
4. 在线坚持度
5. 生活作息稳定性
6. 偏晚消费倾向
7. 健康相关行为与生活方式指数
8. 学业–生活综合风险信号指数
9. 学业–生活方式协同类型
"""
import numpy as np
import pandas as pd
import os
import json
import argparse
PATH_STUDENT_VLE = 'OULAD/studentVle.csv'
PATH_STUDENT_INFO = 'OULAD/studentInfo.csv'
PATH_ASSESSMENTS = 'OULAD/assessments.csv'
PATH_STUDENT_ASSESSMENT = 'OULAD/studentAssessment.csv'
PATH_COURSES = 'OULAD/courses.csv'
PATH_STUDENT_REG = 'OULAD/studentRegistration.csv'
PATH_CAMPUS_CARD = 'CCSD/Data_Campus_card.txt'
PATH_EXAM = 'CCSD/Data_Exam.txt'
PATH_PE = 'dataset/link table_14224.csv'
PATH_STUDENT_LINK = 'dataset/Student_link_generated2.csv'
SEP_CAMPUS_CARD = ','
SEP_EXAM = ','
CAMPUS_STUDENT_ID_COL = 'student_id'
EXAM_STUDENT_ID_COL = 'student_id'
CAMPUS_TIME_COL = 'time'
PE_STUDENT_ID_COL = '学号'
PE_TOTAL_SCORE_COL = '总分(0-120)'
LINK_ID_A_COL = 'StudentID_A'
LINK_ID_B_COL = 'StudentID_B'
LINK_ID_C_COL = 'StudentID_C'
FROZEN_STATS_PATH = 'frozen_thresholds.json'
REF_SIDS = None
FROZEN_STATS = {}
APPLY_ONLY = False
DEFAULT_OUTPUT_TAG = 'GradeU37'

LABEL_METADATA = [
    {
        'id': 1,
        'key': 'online_learning_engagement_trajectory_type',
        'legacy_key': 'trajectory_type',
        'chinese_name': '在线学习参与轨迹类型',
        'english_name': 'Online Learning Engagement Trajectory Type',
        'description': 'Trajectory shape of online learning engagement across course phases.'
    },
    {
        'id': 2,
        'key': 'learning_time_investment_level',
        'legacy_key': 'time_invest_level',
        'chinese_name': '学习时间投入水平',
        'english_name': 'Learning Time Investment Level',
        'description': 'Overall level of online learning time investment and intensity.'
    },
    {
        'id': 3,
        'key': 'academic_procrastination_tendency',
        'legacy_key': 'procrastination_level',
        'chinese_name': '学业拖延倾向',
        'english_name': 'Academic Procrastination Tendency',
        'description': 'Submission timing pattern indicating procrastination tendency.'
    },
    {
        'id': 4,
        'key': 'online_learning_persistence_level',
        'legacy_key': 'persistence_level',
        'chinese_name': '在线坚持度',
        'english_name': 'Online Learning Persistence Level',
        'description': 'Continuity and persistence of online learning engagement.'
    },
    {
        'id': 5,
        'key': 'daily_routine_stability_level',
        'legacy_key': 'routine_stability_level',
        'chinese_name': '生活作息稳定性',
        'english_name': 'Daily Routine Stability Level',
        'description': 'Routine stability inferred from campus card timing distributions.'
    },
    {
        'id': 6,
        'key': 'late_time_consumption_tendency',
        'legacy_key': 'consume_pressure_level',
        'chinese_name': '偏晚消费倾向',
        'english_name': 'Late-Time Consumption Tendency',
        'description': 'Tendency to spend more during later time periods of the day.'
    },
    {
        'id': 7,
        'key': 'health_related_behavior_and_lifestyle_index',
        'legacy_key': 'lifestyle_self_discipline_level',
        'chinese_name': '健康相关行为与生活方式指数',
        'english_name': 'Health-Related Behavior and Lifestyle Index',
        'description': 'Lifestyle index derived from timing regularity and late-time behavior.'
    },
    {
        'id': 8,
        'key': 'academic_life_composite_risk_signal_index',
        'legacy_key': 'stress_risk_level',
        'chinese_name': '学业–生活综合风险信号指数',
        'english_name': 'Academic-Life Composite Risk Signal Index',
        'description': 'Composite signal combining academic risk and lifestyle risk.'
    },
    {
        'id': 9,
        'key': 'academic_lifestyle_synergy_type',
        'legacy_key': 'ach_health_type',
        'chinese_name': '学业–生活方式协同类型',
        'english_name': 'Academic-Lifestyle Synergy Type',
        'description': 'Joint type defined by academic performance and lifestyle profile.'
    },
]

LABEL_KEY_MAP = {item['legacy_key']: item['key'] for item in LABEL_METADATA}
LABEL_NAME_MAP_ZH = {item['key']: item['chinese_name'] for item in LABEL_METADATA}
LABEL_NAME_MAP_EN = {item['key']: item['english_name'] for item in LABEL_METADATA}
LABEL_DESCRIPTION_MAP = {item['key']: item['description'] for item in LABEL_METADATA}
LABEL_EXPORT_COLUMNS = [item['key'] for item in LABEL_METADATA]


def set_ref_sids(ref_sids):
    """Register the reference student ids used to fit frozen statistics."""
    global REF_SIDS
    if ref_sids is None:
        REF_SIDS = None
        return
    REF_SIDS = set((int(x) for x in list(ref_sids)))

def _ref_mask(df: pd.DataFrame, sid_col: str='sid'):
    """Return a boolean mask that selects rows belonging to the reference split."""
    if REF_SIDS is None or sid_col not in df.columns:
        return None
    sid = pd.to_numeric(df[sid_col], errors='coerce')
    return sid.isin(list(REF_SIDS))

def _safe_mean_std(s: pd.Series):
    """Compute a stable mean and standard deviation pair for z-score scaling."""
    s = pd.to_numeric(s, errors='coerce')
    mu = float(s.mean())
    sigma = float(s.std(ddof=0))
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    return (mu, sigma)

def load_frozen_stats(path=None):
    """Load previously saved train-only statistics from disk."""
    global FROZEN_STATS, FROZEN_STATS_PATH
    if path is not None:
        FROZEN_STATS_PATH = path
    if os.path.exists(FROZEN_STATS_PATH):
        with open(FROZEN_STATS_PATH, 'r', encoding='utf-8') as f:
            FROZEN_STATS = json.load(f)
    else:
        FROZEN_STATS = {}

def save_frozen_stats(path=None):
    """Persist fitted frozen statistics to disk for reuse in later runs."""
    global FROZEN_STATS, FROZEN_STATS_PATH
    if path is not None:
        FROZEN_STATS_PATH = path
    with open(FROZEN_STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(FROZEN_STATS, f, ensure_ascii=False, indent=2, sort_keys=True)

def frozen_zscore(df: pd.DataFrame, col: str, sid_col='sid', key=None) -> pd.Series:
    """Apply train-only z-score normalization and reuse frozen parameters when available."""
    global APPLY_ONLY
    if key is None:
        key = f'z::{col}'
    x_all = pd.to_numeric(df[col], errors='coerce')
    if key in FROZEN_STATS and all((k in FROZEN_STATS[key] for k in ['mu', 'sigma'])):
        mu = float(FROZEN_STATS[key]['mu'])
        sigma = float(FROZEN_STATS[key]['sigma'])
    else:
        if APPLY_ONLY:
            raise KeyError(f"Missing frozen parameter {key}. In --apply_only mode, fitting on a new cohort is not allowed.")
        mask = _ref_mask(df, sid_col=sid_col)
        base = x_all[mask] if mask is not None else x_all
        mu, sigma = _safe_mean_std(base)
        FROZEN_STATS[key] = {'mu': mu, 'sigma': sigma}
    z = (x_all - mu) / sigma
    return z.fillna(0.0)

def frozen_median(df: pd.DataFrame, col: str, sid_col='sid', key=None) -> float:
    """Return a median threshold fitted on the reference split only."""
    global APPLY_ONLY
    if key is None:
        key = f'median::{col}'
    x_all = pd.to_numeric(df[col], errors='coerce')
    if key in FROZEN_STATS and 'median' in FROZEN_STATS[key]:
        med = float(FROZEN_STATS[key]['median'])
    else:
        if APPLY_ONLY:
            raise KeyError(f"Missing frozen parameter {key}. In --apply_only mode, fitting on a new cohort is not allowed.")
        mask = _ref_mask(df, sid_col=sid_col)
        base = x_all[mask] if mask is not None else x_all
        med = float(base.median())
        FROZEN_STATS[key] = {'median': med}
    return med

def frozen_quartiles(df: pd.DataFrame, col: str, sid_col='sid', key=None):
    """Return quartile thresholds fitted on the reference split only."""
    global APPLY_ONLY
    if key is None:
        key = f'quartile::{col}'
    x_all = pd.to_numeric(df[col], errors='coerce')
    if key in FROZEN_STATS and all((k in FROZEN_STATS[key] for k in ['q1', 'q2', 'q3'])):
        q1 = float(FROZEN_STATS[key]['q1'])
        q2 = float(FROZEN_STATS[key]['q2'])
        q3 = float(FROZEN_STATS[key]['q3'])
    else:
        if APPLY_ONLY:
            raise KeyError(f"Missing frozen parameter {key}. In --apply_only mode, fitting on a new cohort is not allowed.")
        mask = _ref_mask(df, sid_col=sid_col)
        base = x_all[mask] if mask is not None else x_all
        qs = base.quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = (float(qs.loc[0.25]), float(qs.loc[0.5]), float(qs.loc[0.75]))
        FROZEN_STATS[key] = {'q1': q1, 'q2': q2, 'q3': q3}
    return (q1, q2, q3)

def _stable_jitter_from_sid(df: pd.DataFrame, sid_col: str='sid') -> pd.Series:
    """Generate deterministic tiny noise from sid values for reproducible tie breaking."""
    if sid_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    sid = pd.to_numeric(df[sid_col], errors='coerce').fillna(-1).astype('int64')
    x = sid * 1103515245 + 12345 & 2147483647
    return (x.astype('float64') / float(2147483647)).rename('_jitter')

def add_index_and_level(df, col, new_index_col, new_level_col, sid_col='sid', key=None):
    """Convert a continuous feature into a sigmoid index and a frozen four-level label.

    The z-score parameters and quartile cut points are fitted on the train split only.
    """
    if key is None:
        key = f'bin::{new_level_col}'
    x_all = pd.to_numeric(df[col], errors='coerce')
    if key in FROZEN_STATS and all((k in FROZEN_STATS[key] for k in ['mu', 'sigma', 'q1', 'q2', 'q3'])):
        mu = float(FROZEN_STATS[key]['mu'])
        sigma = float(FROZEN_STATS[key]['sigma'])
        q1 = float(FROZEN_STATS[key]['q1'])
        q2 = float(FROZEN_STATS[key]['q2'])
        q3 = float(FROZEN_STATS[key]['q3'])
    else:
        if APPLY_ONLY:
            raise KeyError(f"Missing frozen parameter {key}. In --apply_only mode, fitting on a new cohort is not allowed.")
        mask = _ref_mask(df, sid_col=sid_col)
        base = x_all[mask] if mask is not None else x_all
        mu, sigma = _safe_mean_std(base)
        z = (x_all - mu) / sigma
        z = z.fillna(0.0)
        idx = 1.0 / (1.0 + np.exp(-z))
        idx_base = idx[mask] if mask is not None else idx
        qs = idx_base.quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = (float(qs.loc[0.25]), float(qs.loc[0.5]), float(qs.loc[0.75]))
        FROZEN_STATS[key] = {'mu': mu, 'sigma': sigma, 'q1': q1, 'q2': q2, 'q3': q3}
    z = (x_all - mu) / sigma
    z = z.fillna(0.0)
    df[new_index_col] = 1.0 / (1.0 + np.exp(-z))
    v = df[new_index_col].fillna(0.5)
    df[new_level_col] = np.where(v < q1, 0, np.where(v < q2, 1, np.where(v < q3, 2, 3))).astype(int)
    return df

def add_index_and_level_zero_bucket(df, col, new_index_col, new_level_col, zero_mask_col, sid_col='sid', key=None):
    """Discretize a zero-inflated feature with an explicit zero bucket.

    Level 0 is reserved for zero or missing activity defined by `zero_mask_col`. The
    remaining non-zero samples are divided into three frozen bins using train-only
    tertile thresholds.
    """
    if key is None:
        key = f'bin0::{new_level_col}'
    x_all = pd.to_numeric(df[col], errors='coerce')
    zero_mask = pd.to_numeric(df[zero_mask_col], errors='coerce').fillna(0) <= 0
    if key in FROZEN_STATS and all((k in FROZEN_STATS[key] for k in ['mu', 'sigma', 'q1', 'q2'])):
        mu = float(FROZEN_STATS[key]['mu'])
        sigma = float(FROZEN_STATS[key]['sigma'])
        q1 = float(FROZEN_STATS[key]['q1'])
        q2 = float(FROZEN_STATS[key]['q2'])
    else:
        if APPLY_ONLY:
            raise KeyError(f"Missing frozen parameter {key}. In --apply_only mode, fitting on a new cohort is not allowed.")
        mask = _ref_mask(df, sid_col=sid_col)
        base = x_all[mask] if mask is not None else x_all
        mu, sigma = _safe_mean_std(base)
        z = (x_all - mu) / sigma
        z = z.fillna(0.0)
        idx = 1.0 / (1.0 + np.exp(-z))
        if mask is not None:
            nz = ~zero_mask & mask
        else:
            nz = ~zero_mask
        idx_base = idx[nz].dropna()
        if len(idx_base) == 0:
            q1, q2 = (1 / 3, 2 / 3)
        else:
            q1 = float(idx_base.quantile(1 / 3))
            q2 = float(idx_base.quantile(2 / 3))
        FROZEN_STATS[key] = {'mu': mu, 'sigma': sigma, 'q1': q1, 'q2': q2}
    z = (x_all - mu) / sigma
    z = z.fillna(0.0)
    df[new_index_col] = 1.0 / (1.0 + np.exp(-z))
    v = df[new_index_col].fillna(0.5)
    df[new_level_col] = 0
    df.loc[~zero_mask & (v < q1), new_level_col] = 1
    df.loc[~zero_mask & (v >= q1) & (v < q2), new_level_col] = 2
    df.loc[~zero_mask & (v >= q2), new_level_col] = 3
    df[new_level_col] = df[new_level_col].astype(int)
    return df

def _read_sid_list(path: str):
    """Read a saved sid split file."""
    s = pd.read_csv(path, header=None)
    return s.iloc[:, 0].dropna().astype(int).unique()

def _write_sid_list(path: str, sids):
    """Write a sid split file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for x in sids:
            f.write(f'{int(x)}\n')

def make_or_load_splits(valid_sids, split_dir='splits', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, force_resplit=False):
    """Create or reuse persistent train/validation/test sid splits."""
    split_dir = split_dir or 'splits'
    os.makedirs(split_dir, exist_ok=True)
    p_train = os.path.join(split_dir, 'train_sids_U.csv')
    p_val = os.path.join(split_dir, 'val_sids_U.csv')
    p_test = os.path.join(split_dir, 'test_sids_U.csv')
    if not force_resplit and os.path.exists(p_train) and os.path.exists(p_val) and os.path.exists(p_test):
        return (_read_sid_list(p_train), _read_sid_list(p_val), _read_sid_list(p_test))
    valid_sids = np.array(list(set((int(x) for x in valid_sids))))
    rng = np.random.RandomState(int(seed))
    rng.shuffle(valid_sids)
    n = len(valid_sids)
    if n == 0:
        return (np.array([]), np.array([]), np.array([]))
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = (train_ratio / total, val_ratio / total, test_ratio / total)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train = max(1, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
    train = valid_sids[:n_train]
    val = valid_sids[n_train:n_train + n_val]
    test = valid_sids[n_train + n_val:]
    _write_sid_list(p_train, train)
    _write_sid_list(p_val, val)
    _write_sid_list(p_test, test)
    return (train, val, test)

def sid_to_split_map(train_sids, val_sids, test_sids):
    """Build a sid-to-split lookup dictionary."""
    m = {}
    for x in train_sids:
        m[int(x)] = 'train'
    for x in val_sids:
        m[int(x)] = 'val'
    for x in test_sids:
        m[int(x)] = 'test'
    return m

def assign_segment(date_array: np.ndarray, length_array: np.ndarray) -> np.ndarray:
    """Map each course day to one of 12 normalized course segments."""
    d = np.array(date_array, dtype=float)
    L = np.array(length_array, dtype=float)
    d = np.where(d < 1, 1, d)
    d = np.where(d > L, L, d)
    seg_width = L / 12.0
    seg = np.floor((d - 1) / seg_width) + 1
    seg = seg.astype(int)
    seg = np.clip(seg, 1, 12)
    return seg

def assign_period(hour: int) -> int:
    """Map an hour of day to one of six time periods."""
    if 0 <= hour < 4:
        return 1
    elif 4 <= hour < 8:
        return 2
    elif 8 <= hour < 12:
        return 3
    elif 12 <= hour < 16:
        return 4
    elif 16 <= hour < 20:
        return 5
    else:
        return 6

def load_data():
    """Load and lightly clean all source datasets used by the pipeline.

    The script expects the same folder structure as the original project:
    `OULAD/`, `CCSD/`, and `dataset/`.
    """
    student_vle = pd.read_csv(PATH_STUDENT_VLE)
    student_info = pd.read_csv(PATH_STUDENT_INFO)
    assessments = pd.read_csv(PATH_ASSESSMENTS)
    student_assessment = pd.read_csv(PATH_STUDENT_ASSESSMENT)
    courses = pd.read_csv(PATH_COURSES)
    student_registration = pd.read_csv(PATH_STUDENT_REG)
    campus_card = pd.read_csv(PATH_CAMPUS_CARD, sep=SEP_CAMPUS_CARD, quotechar="'", skipinitialspace=True, on_bad_lines='skip')
    campus_card.columns = campus_card.columns.str.strip()
    campus_card = campus_card.rename(columns={'Student ID': 'student_id', 'Time': 'time', 'Amount': 'amount'})
    campus_card['student_id'] = campus_card['student_id'].astype(str).str.strip()
    campus_card['time'] = pd.to_datetime(campus_card['time'], errors='coerce')
    campus_card['amount'] = pd.to_numeric(campus_card['amount'], errors='coerce')
    exam = pd.read_csv(PATH_EXAM, sep=SEP_EXAM, header=None, engine='python', on_bad_lines='skip')
    if exam.shape[1] >= 8:
        exam = exam.iloc[:, :8].copy()
    else:
        exam = exam.reindex(columns=list(range(8)))
    exam.columns = ['year', 'semester', 'course_id', 'course_name', 'student_id', 'exam_round', 'course_credit', 'exam_score']
    for col in exam.columns:
        if exam[col].dtype == 'object':
            exam[col] = exam[col].astype(str).str.strip().str.strip('\'"')
    exam['exam_score'] = pd.to_numeric(exam['exam_score'], errors='coerce')
    exam['course_credit'] = pd.to_numeric(exam['course_credit'], errors='coerce')
    try:
        pe = pd.read_csv(PATH_PE)
    except Exception as e:
        print(f'[WARN] Physical-test data was not loaded and will be skipped: {e}')
        pe = pd.DataFrame()
    link = pd.read_csv(PATH_STUDENT_LINK)
    return (student_vle, student_info, assessments, student_assessment, courses, student_registration, campus_card, exam, pe, link)

def build_sid_mapping(student_vle, student_info, student_assessment, courses, student_registration, campus_card, exam, pe, link):
    """Build a shared integer student identifier (`sid`) across all data sources.

    The linkage table connects three id spaces:
    - StudentID_A: CCSD student id
    - StudentID_B: OULAD student id
    - StudentID_C: downstream export id used in the final CSV
    """
    link = link.copy()
    link['StudentID_A_norm'] = link[LINK_ID_A_COL].astype(str).str[-10:]
    link['StudentID_B_norm'] = link[LINK_ID_B_COL].astype(str)
    link['StudentID_C_norm'] = link[LINK_ID_C_COL].astype(str)
    for df in [student_info, student_vle, student_assessment, student_registration]:
        df['id_student'] = df['id_student'].astype(str)
    link['sid'] = np.arange(len(link), dtype=int)
    student_info = student_info.merge(link[['StudentID_B_norm', 'sid']], left_on='id_student', right_on='StudentID_B_norm', how='left')
    student_vle = student_vle.merge(link[['StudentID_B_norm', 'sid']], left_on='id_student', right_on='StudentID_B_norm', how='left')
    student_assessment = student_assessment.merge(link[['StudentID_B_norm', 'sid']], left_on='id_student', right_on='StudentID_B_norm', how='left')
    student_registration = student_registration.merge(link[['StudentID_B_norm', 'sid']], left_on='id_student', right_on='StudentID_B_norm', how='left')
    campus_card['student_id_norm'] = campus_card[CAMPUS_STUDENT_ID_COL].astype(str).str[-10:]
    exam['student_id_norm'] = exam[EXAM_STUDENT_ID_COL].astype(str).str[-10:]
    campus_card = campus_card.merge(link[['StudentID_A_norm', 'sid']], left_on='student_id_norm', right_on='StudentID_A_norm', how='left')
    exam = exam.merge(link[['StudentID_A_norm', 'sid']], left_on='student_id_norm', right_on='StudentID_A_norm', how='left')
    if pe is not None and len(pe) > 0 and (PE_STUDENT_ID_COL in pe.columns):
        pe['学号_norm'] = pe[PE_STUDENT_ID_COL].astype(str)
        pe = pe.merge(link[['StudentID_C_norm', 'sid']], left_on='学号_norm', right_on='StudentID_C_norm', how='left')
    else:
        pe = pd.DataFrame({'sid': pd.Series(dtype='float64')})
    return (student_vle, student_info, student_assessment, courses, student_registration, campus_card, exam, pe, link)

def compute_persistence_level(student_registration, student_info, courses, student_vle, link):
    """Label 4: engagement continuity / persistence.

    This label emphasizes continuity of online engagement over the full registered
    course span. It combines active-day ratio, active-week ratio, recency of the
    last active day, and inactivity gaps. A zero-inflated discretizer is used so
    students with no active weeks are assigned to the lowest level explicitly.
    """
    reg = student_registration.merge(courses[['code_module', 'code_presentation', 'length']], on=['code_module', 'code_presentation'], how='left').merge(student_info[['id_student', 'code_module', 'code_presentation', 'final_result', 'sid']], on=['id_student', 'code_module', 'code_presentation'], how='left')
    if 'sid' not in reg.columns:
        if 'sid_x' in reg.columns:
            reg['sid'] = reg['sid_x']
        elif 'sid_y' in reg.columns:
            reg['sid'] = reg['sid_y']
    for col in ['sid_x', 'sid_y']:
        if col in reg.columns:
            reg.drop(columns=[col], inplace=True)
    reg['date_unregistration_filled'] = reg['date_unregistration'].fillna(reg['length'])
    reg['length'] = pd.to_numeric(reg['length'], errors='coerce').fillna(0)
    reg_start = pd.to_numeric(reg['date_registration'], errors='coerce').fillna(0)
    reg_end = pd.to_numeric(reg['date_unregistration_filled'], errors='coerce').fillna(reg['length'])
    reg_start = reg_start.clip(lower=0)
    reg_end = reg_end.clip(lower=0)
    reg_start = np.minimum(reg_start, reg['length'])
    reg_end = np.minimum(reg_end, reg['length'])
    reg['reg_start'] = reg_start
    reg['reg_end'] = reg_end
    reg['reg_span'] = (reg['reg_end'] - reg['reg_start']).replace(0, 1)
    reg['reg_span'] = reg['reg_span'].clip(lower=1)
    vle_course = student_vle[~student_vle['sid'].isna()].copy().groupby(['sid', 'code_module', 'code_presentation', 'date'], as_index=False)['sum_click'].sum()

    def _max_gap_days(dates):
        dates = np.sort(pd.Series(dates).dropna().unique())
        if len(dates) < 2:
            return np.nan
        gaps = np.diff(dates) - 1
        return float(np.max(gaps)) if len(gaps) else np.nan

    def _reactivation_cnt(dates, gap_thresh=7):
        dates = np.sort(pd.Series(dates).dropna().unique())
        if len(dates) < 2:
            return 0
        gaps = np.diff(dates) - 1
        return int(np.sum(gaps > gap_thresh))
    vle_span = vle_course.groupby(['sid', 'code_module', 'code_presentation']).agg(first_active=('date', 'min'), last_active=('date', 'max'), active_days=('date', 'nunique'), active_weeks=('date', lambda s: s.dropna().astype(int).floordiv(7).nunique()), max_gap_days=('date', _max_gap_days), reactivation_cnt=('date', _reactivation_cnt)).reset_index()
    vle_span['active_span'] = vle_span['last_active'] - vle_span['first_active']
    pers = reg.merge(vle_span, on=['sid', 'code_module', 'code_presentation'], how='left')
    pers['active_days'] = pers['active_days'].fillna(0)
    pers['active_weeks'] = pers['active_weeks'].fillna(0)
    pers['active_span'] = pers['active_span'].fillna(0)
    pers['max_gap_days'] = pers['max_gap_days'].fillna(np.nan)
    pers['reactivation_cnt'] = pers['reactivation_cnt'].fillna(0)
    pers.loc[pers['active_days'] < 2, 'max_gap_days'] = pers.loc[pers['active_days'] < 2, 'reg_span']
    pers['active_days_ratio'] = pers['active_days'] / pers['reg_span']
    pers['active_span_ratio'] = pers['active_span'] / pers['reg_span']
    pers['weeks_span'] = np.ceil(pers['reg_span'] / 7.0).replace(0, 1)
    pers['active_weeks_ratio'] = pers['active_weeks'] / pers['weeks_span']
    pers['last_active_ratio'] = 0.0
    has_act = pers['last_active'].notna()
    pers.loc[has_act, 'last_active_ratio'] = ((pers.loc[has_act, 'last_active'] - pers.loc[has_act, 'reg_start']) / pers.loc[has_act, 'reg_span']).clip(lower=0, upper=1)
    pers['max_gap_ratio'] = (pers['max_gap_days'] / pers['reg_span']).clip(lower=0)
    pers['completed'] = pers['final_result'].isin(['Pass', 'Distinction']).astype(int)
    pers['withdraw'] = pers['final_result'].isin(['Withdrawn']).astype(int)
    if len(pers) == 0:
        return pd.DataFrame(columns=['sid', 'persistence_level', 'persistence_index_sid'])
    for col in ['active_days_ratio', 'active_weeks_ratio', 'last_active_ratio', 'max_gap_ratio']:
        pers['z_' + col] = frozen_zscore(pers, col, sid_col='sid', key=f'z::persistence_v2::{col}')
    pers['pers_raw'] = 0.2 * pers['z_active_days_ratio'] + 0.45 * pers['z_active_weeks_ratio'] + 0.35 * pers['z_last_active_ratio'] - 0.4 * pers['z_max_gap_ratio']
    pers = add_index_and_level_zero_bucket(pers, col='pers_raw', new_index_col='persistence_index', new_level_col='persistence_level_course', zero_mask_col='active_weeks', sid_col='sid', key='bin0::persistence_level_course_v2')
    pers_student = pers.groupby('sid').agg(persistence_index=('persistence_index', 'mean'), active_weeks=('active_weeks', 'sum'), active_days=('active_days', 'sum'), active_span=('active_span', 'mean'), max_gap_days=('max_gap_days', 'mean'), last_active=('last_active', 'max'), completed=('completed', 'max'), withdraw=('withdraw', 'max')).reset_index()
    pers_student = add_index_and_level_zero_bucket(pers_student, col='persistence_index', new_index_col='persistence_index_sid', new_level_col='persistence_level', zero_mask_col='active_weeks', sid_col='sid', key='bin0::persistence_level_sid_v2')
    try:
        raw_by_sc = pers.merge(link[['sid', 'StudentID_C_norm']], on='sid', how='left')[['StudentID_C_norm', 'active_days', 'active_span', 'active_weeks', 'max_gap_days', 'reactivation_cnt', 'last_active', 'completed', 'withdraw']]
        raw_by_sc = raw_by_sc.rename(columns={'StudentID_C_norm': 'StudentID_C'})
        out_path = 'student_persistence_raw_features_by_StudentID_C.csv'
        raw_by_sc.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'[OK] Exported persistence raw features: {os.path.abspath(out_path)}')
    except Exception as e:
        print(f'[WARN] Failed to export persistence raw features: {e}')
    return pers_student[['sid', 'persistence_level', 'persistence_index_sid']]

def compute_routine_period_ratio_by_sid(campus_card):
    """Aggregate campus-card consumption into six daily time periods per student.

    The function returns both amount ratios and frequency ratios. Only negative
    amounts are treated as spending events.
    """
    campus = campus_card[~campus_card['sid'].isna()].copy()
    if len(campus) == 0:
        return pd.DataFrame(columns=['sid'] + [f'amount_ratio_p{i}' for i in range(1, 7)] + [f'freq_ratio_p{i}' for i in range(1, 7)])
    if CAMPUS_TIME_COL not in campus.columns:
        for cand in ['meal time', 'meal_time', '交易时间', 'time']:
            if cand in campus.columns:
                campus.rename(columns={cand: CAMPUS_TIME_COL}, inplace=True)
                break
    campus[CAMPUS_TIME_COL] = pd.to_datetime(campus[CAMPUS_TIME_COL], errors='coerce')
    campus = campus.dropna(subset=[CAMPUS_TIME_COL])
    p1_start = pd.to_datetime('2017-09-01')
    p1_end = pd.to_datetime('2018-02-01')
    p2_start = pd.to_datetime('2018-03-11')
    p2_end = pd.to_datetime('2018-06-30')
    mask = (campus[CAMPUS_TIME_COL] >= p1_start) & (campus[CAMPUS_TIME_COL] <= p1_end) | (campus[CAMPUS_TIME_COL] >= p2_start) & (campus[CAMPUS_TIME_COL] <= p2_end)
    campus = campus[mask].copy()
    if len(campus) == 0:
        return pd.DataFrame(columns=['sid'] + [f'amount_ratio_p{i}' for i in range(1, 7)] + [f'freq_ratio_p{i}' for i in range(1, 7)])
    campus['hour'] = pd.to_datetime(campus[CAMPUS_TIME_COL]).dt.hour
    campus['period'] = campus['hour'].apply(assign_period)
    campus['amount'] = pd.to_numeric(campus['amount'], errors='coerce')
    campus = campus.dropna(subset=['amount'])
    campus['is_consume'] = (campus['amount'] < 0).astype(int)
    campus['consume_amount'] = campus['amount'].where(campus['amount'] < 0, 0.0)
    gp = campus.groupby(['sid', 'period'])
    sid_period = gp.agg(amount_consume=('consume_amount', 'sum'), freq_consume=('is_consume', 'sum')).reset_index()
    sid_total = sid_period.groupby('sid').agg(total_consume_amount=('amount_consume', lambda x: x.abs().sum()), total_consume_freq=('freq_consume', 'sum')).reset_index()
    sid_period = sid_period.merge(sid_total, on='sid', how='left')
    sid_period['amount_ratio'] = sid_period['amount_consume'].abs() / sid_period['total_consume_amount'].replace(0, np.nan)
    sid_period['freq_ratio'] = sid_period['freq_consume'] / sid_period['total_consume_freq'].replace(0, np.nan)
    sid_period[['amount_ratio', 'freq_ratio']] = sid_period[['amount_ratio', 'freq_ratio']].fillna(0.0)
    amt_wide = sid_period.pivot_table(index='sid', columns='period', values='amount_ratio', fill_value=0.0)
    amt_wide.columns = [f'amount_ratio_p{int(c)}' for c in amt_wide.columns]
    freq_wide = sid_period.pivot_table(index='sid', columns='period', values='freq_ratio', fill_value=0.0)
    freq_wide.columns = [f'freq_ratio_p{int(c)}' for c in freq_wide.columns]
    result = pd.concat([amt_wide, freq_wide], axis=1)
    for i in range(1, 7):
        col_a = f'amount_ratio_p{i}'
        col_f = f'freq_ratio_p{i}'
        if col_a not in result.columns:
            result[col_a] = 0.0
        if col_f not in result.columns:
            result[col_f] = 0.0
    result = result.reset_index()
    result = result[['sid'] + [f'amount_ratio_p{i}' for i in range(1, 7)] + [f'freq_ratio_p{i}' for i in range(1, 7)]]
    return result

def compute_routine_stability_level(campus_card):
    """Label 5: routine regularity from timing distribution.

    A student is considered more regular when spending frequency is concentrated in
    fewer time periods. The core signal is the negative entropy of the six-period
    spending-frequency distribution.
    """
    period_df = compute_routine_period_ratio_by_sid(campus_card)
    if len(period_df) == 0:
        return pd.DataFrame(columns=['sid', 'routine_stability_level'])
    freq_cols = [f'freq_ratio_p{i}' for i in range(1, 7)]
    eps = 1e-08

    def calc_entropy(row):
        p = row[freq_cols].values.astype(float)
        s = p.sum()
        if s <= 0:
            return 1.0
        p = p / s
        ent = -(p * np.log(p + eps)).sum() / np.log(6.0)
        return float(ent)
    period_df['freq_entropy'] = period_df.apply(calc_entropy, axis=1)
    period_df['routine_stability_raw'] = -period_df['freq_entropy']
    period_df = add_index_and_level(period_df, col='routine_stability_raw', new_index_col='routine_stability_index', new_level_col='routine_stability_level')
    return period_df[['sid', 'routine_stability_level']]

def compute_consume_pressure_level(campus_card):
    """Label 6: late-time spending tendency.

    Later time periods receive larger weights. Students whose spending is skewed to
    late afternoon and evening obtain higher scores.
    """
    period_df = compute_routine_period_ratio_by_sid(campus_card)
    if len(period_df) == 0:
        return pd.DataFrame(columns=['sid', 'consume_pressure_level'])
    amt_cols = [f'amount_ratio_p{i}' for i in range(1, 7)]
    weights = np.arange(1, 7, dtype=float)

    def calc_pressure(row):
        a = row[amt_cols].values.astype(float)
        s = a.sum()
        if s <= 0:
            return 0.0
        a = a / s
        return float(np.dot(a, weights))
    period_df['consume_pressure_raw'] = period_df.apply(calc_pressure, axis=1)
    period_df = add_index_and_level(period_df, col='consume_pressure_raw', new_index_col='consume_pressure_index', new_level_col='consume_pressure_level')
    return period_df[['sid', 'consume_pressure_level']]

def convert_sid_to_studentC(period_ratio_file='routine_period_ratio_by_sid.csv', link_file=PATH_STUDENT_LINK, output_file='routine_period_ratio_by_StudentC.csv'):
    """Convert a sid-based routine-period export into a StudentID_C-based export."""
    df = pd.read_csv(period_ratio_file, encoding='utf-8-sig')
    link = pd.read_csv(link_file)
    link = link.copy()
    link['StudentID_A_norm'] = link[LINK_ID_A_COL].astype(str).str[-10:]
    link['StudentID_B_norm'] = link[LINK_ID_B_COL].astype(str)
    link['StudentID_C_norm'] = link[LINK_ID_C_COL].astype(str)
    link['sid'] = np.arange(len(link), dtype=int)
    df['sid'] = df['sid'].astype('Int64')
    link['sid'] = link['sid'].astype('Int64')
    merged = df.merge(link[['sid', 'StudentID_C_norm']], on='sid', how='left')
    merged = merged[~merged['StudentID_C_norm'].isna()].copy()
    merged = merged.rename(columns={'StudentID_C_norm': 'StudentID_C'})
    cols = merged.columns.tolist()
    cols.remove('StudentID_C')
    if 'sid' in cols:
        cols.remove('sid')
    cols = ['StudentID_C'] + cols
    merged = merged[cols]
    merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'[OK] Exported StudentID_C routine period ratios: {os.path.abspath(output_file)}')

def compute_time_invest_level(student_vle):
    """Label 2: engagement intensity / time investment.

    The label summarizes how much a student clicks, how many active days they have,
    how intense activity is on active days, and how variable that activity is.
    """
    daily = student_vle[~student_vle['sid'].isna()].copy().groupby(['sid', 'code_module', 'code_presentation', 'date'], as_index=False)['sum_click'].sum()

    def compute_time_features(g):
        clicks = g['sum_click'].values
        total_clicks = float(clicks.sum())
        active_days = int((clicks > 0).sum())
        clicks_per_active = total_clicks / (active_days + 1e-06)
        mean = clicks.mean()
        std = clicks.std(ddof=0)
        cv = std / (mean + 1e-06)
        return pd.Series({'total_clicks': total_clicks, 'active_days': active_days, 'clicks_per_active': clicks_per_active, 'click_cv': cv})
    time_df = daily.groupby(['sid', 'code_module', 'code_presentation']).apply(compute_time_features).reset_index()
    if len(time_df) == 0:
        return pd.DataFrame(columns=['sid', 'time_invest_level'])
    for col in ['total_clicks', 'active_days', 'clicks_per_active', 'click_cv']:
        time_df['z_' + col] = frozen_zscore(time_df, col, sid_col='sid', key=f'z::time_invest::{col}')
    time_df['time_raw'] = time_df['z_total_clicks'] + time_df['z_active_days'] + time_df['z_clicks_per_active'] - time_df['z_click_cv']
    time_df = add_index_and_level(time_df, col='time_raw', new_index_col='time_invest_index', new_level_col='time_invest_level_course')
    time_student = time_df.groupby('sid')['time_invest_index'].mean().reset_index()
    time_student = add_index_and_level(time_student, col='time_invest_index', new_index_col='time_invest_index_sid', new_level_col='time_invest_level')
    return time_student[['sid', 'time_invest_level', 'time_invest_index_sid']]

def compute_procrastination_level(assessments, student_assessment):
    """Label 3: submission timing / procrastination pattern.

    The label uses lead time before deadlines and the ratio of late submissions. A
    larger value indicates stronger procrastination risk.
    """
    ass = assessments[['id_assessment', 'code_module', 'code_presentation', 'date']].rename(columns={'date': 'deadline'})
    stu_ass = student_assessment[['id_assessment', 'id_student', 'date_submitted', 'score', 'sid']]
    ass_merged = stu_ass.merge(ass, on='id_assessment', how='left')
    ass_merged = ass_merged[~ass_merged['sid'].isna()].copy()
    if len(ass_merged) == 0:
        return pd.DataFrame(columns=['sid', 'procrastination_level'])
    ass_merged['lead_time'] = ass_merged['deadline'] - ass_merged['date_submitted']

    def compute_procr(g):
        if len(g) == 0:
            return pd.Series({'avg_lead_time': 0.0, 'late_ratio': 0.0})
        avg_lead = float(g['lead_time'].mean())
        late_ratio = float((g['lead_time'] < 0).mean())
        return pd.Series({'avg_lead_time': avg_lead, 'late_ratio': late_ratio})
    procr = ass_merged.groupby('sid').apply(compute_procr).reset_index()
    for col in ['avg_lead_time', 'late_ratio']:
        procr['z_' + col] = frozen_zscore(procr, col, sid_col='sid', key=f'z::procrast::{col}')
    procr['procr_raw'] = -procr['z_avg_lead_time'] + procr['z_late_ratio']
    procr = add_index_and_level(procr, col='procr_raw', new_index_col='procrast_index', new_level_col='procrastination_level')
    return procr[['sid', 'procrastination_level', 'procrast_index', 'procr_raw']]

def compute_trajectory_type(student_vle, courses):
    """Label 1: engagement trajectory shape.

    For each course, click activity is summarized over normalized early and late
    phases. The difference between late and early activity defines a trajectory
    slope, which is then discretized into four levels.
    """
    sv = student_vle[~student_vle['sid'].isna()].copy()
    sv = sv.merge(courses[['code_module', 'code_presentation', 'length']], on=['code_module', 'code_presentation'], how='left')
    sv = sv.dropna(subset=['length'])
    sv['segment'] = assign_segment(sv['date'].values, sv['length'].values)
    seg_clicks = sv.groupby(['sid', 'code_module', 'code_presentation', 'segment'], as_index=False)['sum_click'].sum().rename(columns={'sum_click': 'seg_clicks'})
    traj = seg_clicks.pivot_table(index=['sid', 'code_module', 'code_presentation'], columns='segment', values='seg_clicks', aggfunc='sum', fill_value=0.0).reset_index()
    for seg in range(1, 13):
        if seg not in traj.columns:
            traj[seg] = 0.0
    traj['early'] = traj[[1, 2, 3]].sum(axis=1)
    traj['mid'] = traj[[4, 5, 6]].sum(axis=1)
    traj['late'] = traj[[7, 8, 9]].sum(axis=1)
    traj['end'] = traj[[10, 11, 12]].sum(axis=1)
    traj['phase_early'] = traj['early']
    traj['phase_late'] = traj['late'] + traj['end']
    if len(traj) == 0:
        return pd.DataFrame(columns=['sid', 'trajectory_type_sid', 'slope_mean'])
    traj['slope'] = traj['phase_late'] - traj['phase_early']
    q1, q2, q3 = frozen_quartiles(traj, 'slope', sid_col='sid', key='quartile::trajectory::slope')
    if np.isnan(q1) or np.isnan(q2) or np.isnan(q3):
        qs = pd.to_numeric(traj['slope'], errors='coerce').dropna().quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = (float(qs.loc[0.25]), float(qs.loc[0.5]), float(qs.loc[0.75]))

    def classify_traj(row):
        s = row['slope']
        if s <= q1:
            return 0
        elif s <= q2:
            return 1
        elif s <= q3:
            return 2
        else:
            return 3
    traj['trajectory_type'] = traj.apply(classify_traj, axis=1)
    traj_student = traj.groupby('sid').agg(trajectory_type=('trajectory_type', lambda x: int(round(float(np.mean(x))))), slope_mean=('slope', 'mean')).reset_index().rename(columns={'trajectory_type': 'trajectory_type_sid'})
    return traj_student

def compute_ach_health_type(exam, lifestyle_df, link):
    """Label 9: academic-lifestyle synergy type.

    This label is a 2x2 type constructed from train-frozen medians of GPA and the
    lifestyle score:
    0 = low academic / low lifestyle
    1 = low academic / high lifestyle
    2 = high academic / low lifestyle
    3 = high academic / high lifestyle
    """
    exam_s = exam[~exam['sid'].isna()].copy()
    if 'course_credit' not in exam_s.columns:
        raise ValueError("Data_Exam must contain a 'course_credit' column to compute weighted GPA.")
    if len(exam_s) == 0:
        return pd.DataFrame(columns=['sid', 'ach_health_type'])

    def gpa_func(g):
        return float(np.average(g['exam_score'], weights=g['course_credit']))
    gpa = exam_s.groupby('sid').apply(gpa_func).reset_index(name='gpa_score')
    try:
        if 'StudentID_C_norm' in link.columns:
            gpa_export = gpa.merge(link[['sid', 'StudentID_C_norm']].drop_duplicates(), on='sid', how='left')
            gpa_export = gpa_export[~gpa_export['StudentID_C_norm'].isna()].copy()
            gpa_export = gpa_export.rename(columns={'StudentID_C_norm': 'StudentID_C'})
            gpa_out_path = 'gpa_by_StudentID_C.csv'
            gpa_export[['StudentID_C', 'gpa_score']].to_csv(gpa_out_path, index=False, encoding='utf-8-sig')
            print(f'[OK] Exported GPA file: {os.path.abspath(gpa_out_path)}')
        else:
            print('[WARN] StudentID_C_norm is missing in the linkage table, so GPA cannot be exported by StudentID_C.')
    except Exception as e:
        print(f'[WARN] Failed to export GPA details: {e}')
    if lifestyle_df is None or len(lifestyle_df) == 0:
        return pd.DataFrame(columns=['sid', 'ach_health_type'])
    tmp = lifestyle_df[['sid', 'lifestyle_raw']].copy()
    tmp['lifestyle_raw'] = pd.to_numeric(tmp['lifestyle_raw'], errors='coerce')
    df = gpa.merge(tmp, on='sid', how='inner')
    if len(df) == 0:
        return pd.DataFrame(columns=['sid', 'ach_health_type'])
    gpa_med = frozen_median(df, 'gpa_score', sid_col='sid', key='median::ach_lifestyle::gpa')
    life_med = frozen_median(df, 'lifestyle_raw', sid_col='sid', key='median::ach_lifestyle::lifestyle_raw')

    def classify(row):
        acad_high = row['gpa_score'] >= gpa_med
        life_high = row['lifestyle_raw'] >= life_med
        if not acad_high and (not life_high):
            return 0
        elif not acad_high and life_high:
            return 1
        elif acad_high and (not life_high):
            return 2
        else:
            return 3
    df['ach_health_type'] = df.apply(classify, axis=1)
    return df[['sid', 'ach_health_type']]

def compute_lifestyle_self_discipline_level(campus_card):
    """Label 7: health-related behavior and lifestyle index.

    This open-source version no longer depends on physical-test data. Instead, it
    uses campus-card timing patterns as a proxy for lifestyle regularity, especially
    night-time frequency and late-time spending.
    """
    period_df = compute_routine_period_ratio_by_sid(campus_card)
    if len(period_df) == 0:
        return pd.DataFrame(columns=['sid', 'lifestyle_self_discipline_level', 'lifestyle_raw'])
    freq_cols = [f'freq_ratio_p{i}' for i in range(1, 7)]
    amt_cols = [f'amount_ratio_p{i}' for i in range(1, 7)]
    for col in freq_cols + amt_cols:
        if col not in period_df.columns:
            period_df[col] = 0.0
    eps = 1e-12

    def calc_entropy(row):
        p = row[freq_cols].values.astype(float)
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        return float(-np.sum(p * np.log(p + eps)) / np.log(6.0))
    period_df['freq_entropy'] = period_df.apply(calc_entropy, axis=1)
    period_df['night_freq_ratio'] = period_df['freq_ratio_p1'] + period_df['freq_ratio_p6']
    period_df['late_amount_ratio'] = period_df['amount_ratio_p5'] + period_df['amount_ratio_p6']
    df = period_df[['sid', 'freq_entropy', 'night_freq_ratio', 'late_amount_ratio']].copy()
    df[['freq_entropy', 'night_freq_ratio', 'late_amount_ratio']] = df[['freq_entropy', 'night_freq_ratio', 'late_amount_ratio']].fillna(0.0)
    df['minus_entropy'] = -df['freq_entropy']
    for col, key in [('minus_entropy', 'z::lifestyle::minus_entropy'), ('night_freq_ratio', 'z::lifestyle::night_freq_ratio'), ('late_amount_ratio', 'z::lifestyle::late_amount_ratio')]:
        df['z_' + col] = frozen_zscore(df, col, sid_col='sid', key=key)
    df['lifestyle_raw'] = df['z_minus_entropy'] - df['z_night_freq_ratio'] - df['z_late_amount_ratio']
    df = add_index_and_level(df, col='lifestyle_raw', new_index_col='lifestyle_self_discipline_index', new_level_col='lifestyle_self_discipline_level')
    return df[['sid', 'lifestyle_self_discipline_level', 'lifestyle_raw']]

def compute_stress_risk_level(traj_student, time_student, procr_student, pers_student, campus_card):
    """Label 8: academic-lifestyle composite risk signal.

    The label combines academic-side risk signals, such as procrastination and weak
    persistence, with lifestyle-side risk signals derived from timing entropy and
    late-time behavior.
    """
    acad = None
    for df in [traj_student, time_student, procr_student, pers_student]:
        if df is None or len(df) == 0:
            continue
        if acad is None:
            acad = df.copy()
        else:
            acad = acad.merge(df, on='sid', how='left')
    if acad is None or len(acad) == 0:
        return pd.DataFrame(columns=['sid', 'stress_risk_level'])
    for col in ['slope_mean', 'time_invest_index_sid', 'procrast_index', 'persistence_index_sid']:
        if col not in acad.columns:
            acad[col] = 0.0
        acad[col] = pd.to_numeric(acad[col], errors='coerce').fillna(0.0)
    acad['z_time'] = frozen_zscore(acad, 'time_invest_index_sid', sid_col='sid', key='z::stress::time_invest_index_sid')
    acad['z_pers'] = frozen_zscore(acad, 'persistence_index_sid', sid_col='sid', key='z::stress::persistence_index_sid')
    acad['z_procr'] = frozen_zscore(acad, 'procrast_index', sid_col='sid', key='z::stress::procrast_index')
    acad['z_slope'] = frozen_zscore(acad, 'slope_mean', sid_col='sid', key='z::stress::slope_mean')
    acad['acad_risk_raw'] = -acad['z_time'] - acad['z_pers'] + acad['z_procr'] - acad['z_slope']
    period_df = compute_routine_period_ratio_by_sid(campus_card)
    if len(period_df) == 0:
        life = pd.DataFrame({'sid': acad['sid'].values, 'freq_entropy': 0.0, 'night_freq_ratio': 0.0, 'late_amount_ratio': 0.0})
    else:
        freq_cols = [f'freq_ratio_p{i}' for i in range(1, 7)]
        amt_cols = [f'amount_ratio_p{i}' for i in range(1, 7)]
        for col in freq_cols + amt_cols:
            if col not in period_df.columns:
                period_df[col] = 0.0
        eps = 1e-12

        def calc_entropy(row):
            p = row[freq_cols].values.astype(float)
            s = p.sum()
            if s <= 0:
                return 0.0
            p = p / s
            return float(-np.sum(p * np.log(p + eps)) / np.log(6.0))
        period_df['freq_entropy'] = period_df.apply(calc_entropy, axis=1)
        period_df['night_freq_ratio'] = period_df['freq_ratio_p1'] + period_df['freq_ratio_p6']
        period_df['late_amount_ratio'] = period_df['amount_ratio_p5'] + period_df['amount_ratio_p6']
        life = period_df[['sid', 'freq_entropy', 'night_freq_ratio', 'late_amount_ratio']].copy()
    life[['freq_entropy', 'night_freq_ratio', 'late_amount_ratio']] = life[['freq_entropy', 'night_freq_ratio', 'late_amount_ratio']].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df = acad.merge(life, on='sid', how='left')
    df[['freq_entropy', 'night_freq_ratio', 'late_amount_ratio']] = df[['freq_entropy', 'night_freq_ratio', 'late_amount_ratio']].fillna(0.0)
    df['z_entropy'] = frozen_zscore(df, 'freq_entropy', sid_col='sid', key='z::stress::freq_entropy')
    df['z_night'] = frozen_zscore(df, 'night_freq_ratio', sid_col='sid', key='z::stress::night_freq_ratio')
    df['z_late'] = frozen_zscore(df, 'late_amount_ratio', sid_col='sid', key='z::stress::late_amount_ratio')
    df['life_risk_raw'] = df['z_entropy'] + df['z_night'] + df['z_late']
    df['stress_raw'] = df['acad_risk_raw'] + df['life_risk_raw']
    df = add_index_and_level(df, col='stress_raw', new_index_col='stress_risk_index', new_level_col='stress_risk_level')
    return df[['sid', 'stress_risk_level']]

def export_label_metadata(mapping_csv_path: str, mapping_json_path: str):
    """Export label metadata tables for repository users and downstream analysis."""
    mapping_df = pd.DataFrame(LABEL_METADATA)[['id', 'key', 'legacy_key', 'chinese_name', 'english_name', 'description']]
    mapping_df.to_csv(mapping_csv_path, index=False, encoding='utf-8-sig')
    with open(mapping_json_path, 'w', encoding='utf-8') as file_obj:
        json.dump(LABEL_METADATA, file_obj, ensure_ascii=False, indent=2)


def main(args):
    """Run the full pipeline from raw files to exported label tables.

    The exported main CSV uses stable English label keys suitable for open-source
    code, while separate mapping files preserve the Chinese label names and the
    legacy internal column names used in earlier versions of the project.
    """
    global APPLY_ONLY, FROZEN_STATS_PATH
    APPLY_ONLY = bool(args.apply_only)
    FROZEN_STATS_PATH = args.frozen_stats
    load_frozen_stats(FROZEN_STATS_PATH)
    student_vle, student_info, assessments, student_assessment, courses, student_registration, campus_card, exam, pe, link = load_data()
    student_vle, student_info, student_assessment, courses, student_registration, campus_card, exam, pe, link = build_sid_mapping(student_vle, student_info, student_assessment, courses, student_registration, campus_card, exam, pe, link)
    valid_sids = link['sid'].dropna().astype(int).unique()
    if args.skip_split:
        train_sids, val_sids, test_sids = (valid_sids, np.array([]), np.array([]))
    else:
        train_sids, val_sids, test_sids = make_or_load_splits(valid_sids=valid_sids, split_dir=args.split_dir, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed, force_resplit=args.force_resplit)
    set_ref_sids(train_sids)
    routine_period = compute_routine_period_ratio_by_sid(campus_card)
    routine_period_out = 'routine_period_ratio_by_sid.csv'
    routine_period.to_csv(routine_period_out, index=False, encoding='utf-8-sig')
    print(f'[OK] Exported routine_period_ratio_by_sid: {os.path.abspath(routine_period_out)}')
    traj_student = compute_trajectory_type(student_vle, courses)
    time_student = compute_time_invest_level(student_vle)
    procr_student = compute_procrastination_level(assessments, student_assessment)
    pers_student = compute_persistence_level(student_registration, student_info, courses, student_vle, link)
    routine_student = compute_routine_stability_level(campus_card)
    pressure_student = compute_consume_pressure_level(campus_card)
    lifestyle_student = compute_lifestyle_self_discipline_level(campus_card)
    ach_health_student = compute_ach_health_type(exam, lifestyle_student, link)
    stress_student = compute_stress_risk_level(traj_student, time_student, procr_student, pers_student, campus_card)
    final = link[['sid', 'StudentID_C_norm']].drop_duplicates().copy()
    final = final.merge(traj_student, on='sid', how='left')
    final = final.merge(time_student, on='sid', how='left')
    final = final.merge(procr_student, on='sid', how='left')
    final = final.merge(pers_student, on='sid', how='left')
    final = final.merge(routine_student, on='sid', how='left')
    final = final.merge(pressure_student, on='sid', how='left')
    final = final.merge(ach_health_student, on='sid', how='left')
    final = final.merge(lifestyle_student, on='sid', how='left')
    final = final.merge(stress_student, on='sid', how='left')
    final = final.rename(columns={'StudentID_C_norm': 'StudentID_C'})
    if args.skip_split:
        final['split'] = 'all'
    else:
        split_map = sid_to_split_map(train_sids, val_sids, test_sids)
        final['split'] = final['sid'].map(lambda x: split_map.get(int(x), 'unknown') if pd.notna(x) else 'unknown')
    final = final.rename(columns={'trajectory_type_sid': 'trajectory_type'})
    final = final.rename(columns=LABEL_KEY_MAP)
    cols = ['sid', 'StudentID_C', 'split'] + LABEL_EXPORT_COLUMNS
    for col in cols:
        if col not in final.columns:
            final[col] = np.nan
    final = final[cols]
    export_label_metadata(args.label_mapping_csv, args.label_mapping_json)
    final.to_csv(args.out_all, index=False, encoding='utf-8-sig')
    print(f'[OK] Exported full label table: {os.path.abspath(args.out_all)}')
    print(f'[OK] Exported label mapping CSV: {os.path.abspath(args.label_mapping_csv)}')
    print(f'[OK] Exported label mapping JSON: {os.path.abspath(args.label_mapping_json)}')
    if not args.skip_split:
        final[final['split'] == 'train'].to_csv(args.out_train, index=False, encoding='utf-8-sig')
        final[final['split'] == 'val'].to_csv(args.out_val, index=False, encoding='utf-8-sig')
        final[final['split'] == 'test'].to_csv(args.out_test, index=False, encoding='utf-8-sig')
        print(f'[OK] Exported train split: {os.path.abspath(args.out_train)}')
        print(f'[OK] Exported validation split: {os.path.abspath(args.out_val)}')
        print(f'[OK] Exported test split: {os.path.abspath(args.out_test)}')
    if not APPLY_ONLY:
        save_frozen_stats(FROZEN_STATS_PATH)
        print(f'[OK] Saved frozen statistics: {os.path.abspath(FROZEN_STATS_PATH)}')

def parse_args():
    """Define command-line arguments for reproducible pipeline execution."""
    p = argparse.ArgumentParser(description='Build nine student labels with fixed student splits and train-only frozen discretization.')
    p.add_argument('--frozen_stats', type=str, default='frozen_thresholds.json', help='Path to the frozen-statistics JSON file.')
    p.add_argument('--apply_only', action='store_true', help='Apply existing frozen statistics only. Do not fit missing keys.')
    p.add_argument('--split_dir', type=str, default='splits', help='Directory used to store train/val/test sid split files.')
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=35)
    p.add_argument('--force_resplit', action='store_true', help='Force a new split even if saved split files already exist.')
    p.add_argument('--skip_split', action='store_true', help='Skip splitting, for example when applying frozen statistics to a new cohort.')
    p.add_argument('--out_all', type=str, default=f'student_nine_labels_{DEFAULT_OUTPUT_TAG}.csv')
    p.add_argument('--out_train', type=str, default=f'student_nine_labels_train_{DEFAULT_OUTPUT_TAG}.csv')
    p.add_argument('--out_val', type=str, default=f'student_nine_labels_val_{DEFAULT_OUTPUT_TAG}.csv')
    p.add_argument('--out_test', type=str, default=f'student_nine_labels_test_{DEFAULT_OUTPUT_TAG}.csv')
    p.add_argument('--label_mapping_csv', type=str, default='label_column_mapping.csv', help='Path to the CSV file that stores English-Chinese label column mappings.')
    p.add_argument('--label_mapping_json', type=str, default='label_column_mapping.json', help='Path to the JSON file that stores English-Chinese label metadata.')
    return p.parse_args()
if __name__ == '__main__':
    args = parse_args()
    main(args)
