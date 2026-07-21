from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pymovements as pm
from loguru import logger
from pymovements import ResourceDefinitions

from src.configs.constants import STATS_FOLDER, DataSets
from src.configs.data import get_data_args
from src.data.preprocessing.stats import summarize_dataframe

logger.add('logs/preprocessing.log', level='INFO')


def combine_files(
    dataset: list[pm.reading_measures.ReadingMeasures | pl.DataFrame],
    fileinfo: list[pl.DataFrame],
    output_csv: Path,
    output_summary_csv: Path,
    dataset_name: str,
) -> None:
    # Combine all precomputed events into a single DataFrame
    if len(dataset) > 1:
        logger.info('Merging...')
        combined_df = pd.concat(
            [
                # Combine each frame with its corresponding fileinfo
                pd.concat(
                    [
                        frame.frame.to_pandas().rename(
                            {
                                f: f.replace('"', '')
                                for f in frame.frame.to_pandas().columns
                            },
                            axis=1,
                        ),
                        pd.DataFrame(
                            {
                                'source_file': [
                                    subject_id for _ in range(len(frame.frame))
                                ]
                            }
                        ),
                    ],
                    axis=1,
                )
                for frame, subject_id in zip(dataset, fileinfo['subject_id'])
            ],
            ignore_index=True,
        )
    else:
        logger.info('Only one datafile...')
        combined_df = dataset[0].frame.to_pandas()

    logger.info(f'Total combined rows: {len(combined_df)}')

    # Save combined CSV
    combined_df.to_csv(output_csv, index=False)
    logger.info(f'Combined CSV saved: {output_csv}')

    # Save column summary
    summary_df = summarize_dataframe(combined_df, dataset_name)
    summary_df.to_csv(output_summary_csv, index=False)
    logger.info(f'Column summary saved: {output_summary_csv}')


def combine_stimulus_files(
    data_path: Path,
    matching_pattern: str,
    dataset_name: str,
) -> None:
    """
    Combine stimulus files from a given data path matching a specific pattern.

    Args:
        data_path (str): Path to the directory containing stimulus files.
        matching_pattern (str): Pattern to match files for combining.
    """
    stimulus_files = list(data_path.rglob(matching_pattern))

    if not stimulus_files:
        logger.warning(f'No files found matching {matching_pattern} in {data_path}.')
        return

    combined_df = pd.DataFrame()
    for file_ in stimulus_files:
        if 'reading' in str(file_):
            read_df = pd.read_csv(file_, sep='\t')
            read_df['filename'] = f'{file_.name.split("_")[0]}.png'
            read_df['sequence_num'] = -1
            combined_df = pd.concat([combined_df, read_df])
        else:
            quest_df = pd.read_csv(file_)
            quest_df['sequence_num'] = file_.name.split('_')[0].split('-')[-1]
            combined_df = pd.concat([combined_df, quest_df])

    if 'SBSAT' in str(data_path):
        logger.info('Filling in missing values for SBSAT dataset...')
        combined_df['stimulus_type'] = combined_df['filename'].apply(
            lambda x: x.split('-')[1]
        )
        combined_df['is_question'].fillna(False, inplace=True)

        # Create 'question' column by concatenating all 'word' where 'is_question' is True for each filename
        question_map = (
            combined_df[combined_df['is_question']]
            .groupby(['stimulus_type', 'sequence_num'])['word']
            .apply(' '.join)
        )
        combined_df['question'] = combined_df.apply(
            lambda row: question_map.get(
                (row['stimulus_type'], row['sequence_num']), None
            ),  # type: ignore
            axis=1,
        )

    combined_df.to_csv(Path(data_path) / 'combined_stimulus.csv', index=False)
    logger.info(f'Combined stimulus CSV saved: {data_path / "combined_stimulus.csv"}')


# --- ROAMM ------------------------------------------------------------------
# ROAMM ships as per-subject, per-run sample-level pandas .pkl files (EEG +
# eye-tracking co-registered at 256 Hz) rather than EyeLink-style Interest
# Area / Fixation reports, and it is not in pymovements.DatasetLibrary. This
# section converts the raw .pkl files (already downloaded via download_roamm)
# into combined_ia.csv / combined_fixations.csv in the shape every other
# EyeBench processor expects.

# Every non-EEG column in the ROAMM sample-level .pkl files.
ROAMM_NON_EEG_COLUMNS = [
    'time', 'sfreq', 'first_pass_reading', 'page_num', 'page_start', 'page_end',
    'page_dur', 'is_mw', 'mw_onset', 'mw_offset', 'mw_dur', 'run_num', 'story_name',
    'is_fixation', 'fix_eye', 'fix_tStart', 'fix_tEnd', 'fix_duration', 'fix_xAvg',
    'fix_yAvg', 'fix_pupilAvg', 'fix_fixed_word', 'fix_fixed_word_key',
    'is_blink', 'blink_eye', 'blink_tStart', 'blink_tEnd', 'blink_duration',
    'is_saccade', 'sacc_eye', 'sacc_tStart', 'sacc_tEnd', 'sacc_duration',
    'sacc_xStart', 'sacc_yStart', 'sacc_xEnd', 'sacc_yEnd', 'sacc_ampDeg', 'sacc_vPeak',
    'tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil',
]

ROAMM_STORY_NAMES = {
    'the_voynich_manuscript',
    'prisoners_dilemma',
    'serena_williams',
    'history_of_film',
    'pluto',
}


def _load_all_roamm_runs(subject_ml_root: Path) -> pd.DataFrame:
    """Concatenate every subject's 5 per-run .pkl files, dropping EEG columns
    and non-first-pass-reading samples (task instructions, MW reporting
    screens, re-reads, comprehension questions) to keep memory in check."""
    frames = []
    for subject_dir in sorted(subject_ml_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for pkl_file in sorted(subject_dir.glob('*_ml_data.pkl')):
            df = pd.read_pickle(pkl_file)
            df['participant_id'] = subject_dir.name
            keep_cols = [c for c in ROAMM_NON_EEG_COLUMNS if c in df.columns] + [
                'participant_id'
            ]
            df = df[keep_cols]
            df = df[df['first_pass_reading'] == True]  # noqa: E712
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    unexpected_stories = set(combined['story_name'].unique()) - ROAMM_STORY_NAMES
    if unexpected_stories:
        logger.warning(f'Unexpected ROAMM story_name values: {unexpected_stories}')

    return combined


def _roamm_samples_to_fixation_report(samples: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-sample rows (expanded across each fixation's start/end)
    into one row per fixation event. fix_xAvg/fix_yAvg/fix_pupilAvg are
    already fixation-level averages in the source data."""
    fixations = samples[samples['is_fixation'] == True].copy()  # noqa: E712

    group_keys = ['participant_id', 'story_name', 'page_num', 'fix_tStart', 'fix_tEnd']
    agg = fixations.groupby(group_keys, as_index=False).agg(
        CURRENT_FIX_DURATION=('fix_duration', 'first'),
        CURRENT_FIX_X=('fix_xAvg', 'first'),
        CURRENT_FIX_Y=('fix_yAvg', 'first'),
        CURRENT_FIX_PUPIL=('fix_pupilAvg', 'first'),
        fix_eye=('fix_eye', 'first'),
        fix_fixed_word=('fix_fixed_word', 'first'),
        fix_fixed_word_key=('fix_fixed_word_key', 'first'),
        run_num=('run_num', 'first'),
        is_mw=('is_mw', 'max'),
        page_dur=('page_dur', 'first'),
    )
    agg = agg.sort_values(group_keys).reset_index(drop=True)
    agg['CURRENT_FIX_INDEX'] = (
        agg.groupby(['participant_id', 'story_name', 'page_num']).cumcount() + 1
    )
    return agg


def _load_roamm_stimuli(stimuli_dir: Path) -> pd.DataFrame:
    """Load and concatenate the 5 per-article word-level stimulus/AOI files
    from ROAMM/wiki_stories/*_coordinates.csv."""
    frames = []
    for csv_path in sorted(stimuli_dir.glob('*_coordinates.csv')):
        df = pd.read_csv(csv_path)
        # filenames vary between distributions: some are "<story>_coordinates.csv",
        # others "<story>_control_coordinates.csv" - strip either suffix so
        # story_name always matches the pkl-derived values ('pluto', not
        # 'pluto_control').
        stem = csv_path.name.removesuffix('_coordinates.csv')
        stem = stem.removesuffix('_control')
        df['story_name'] = stem
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f'No *_coordinates.csv stimulus files found under {stimuli_dir}. '
            f'Expected files like pluto_coordinates.csv, history_of_film_coordinates.csv, etc.'
        )

    stim_df = pd.concat(frames, ignore_index=True)

    required_source_columns = ['words', 'word_key', 'page', 'left', 'top', 'width', 'height']
    missing = [c for c in required_source_columns if c not in stim_df.columns]
    if missing:
        raise KeyError(
            f'wiki_stories CSVs are missing expected column(s) {missing}. '
            f'Actual columns found: {list(stim_df.columns)}. '
            f'The real file schema may differ from what this function assumes - '
            f'please share the header row of one *_coordinates.csv file to fix this.'
        )

    if 'is_error' in stim_df.columns:
        stim_df = stim_df[stim_df['is_error'] == 0]

    stim_df = stim_df.rename(
        columns={
            'words': 'IA_LABEL',
            'word_key': 'fix_fixed_word_key',
            'page': 'page_num',
            'left': 'IA_LEFT',
            'top': 'IA_TOP',
            'width': 'IA_WIDTH',
            'height': 'IA_HEIGHT',
            'center_x': 'IA_CENTER_X',
            'center_y': 'IA_CENTER_Y',
        }
    )

    # The pasted sample schema had a running index column named 'Unnamed: 0'
    # (does not reset per page). The real *_coordinates.csv files may not carry
    # that exact column - if so, fall back to on-disk row order, which is
    # already left-to-right/top-to-bottom reading order within each page in
    # every sample we've seen. Either way we end up with a clean, contiguous
    # 0-based per-page word index rather than trusting the raw column values.
    order_col = 'Unnamed: 0' if 'Unnamed: 0' in stim_df.columns else None
    if order_col is None:
        logger.warning(
            "No 'Unnamed: 0' running-index column found in the wiki_stories CSVs - "
            'falling back to on-disk row order per story to determine word order. '
            'Verify this against the real file structure before trusting IA order.'
        )
        stim_df = stim_df.reset_index(drop=True)
        stim_df['_row_order'] = stim_df.index
        order_col = '_row_order'

    stim_df = stim_df.sort_values(['story_name', 'page_num', order_col])
    stim_df['word_index_in_text'] = stim_df.groupby(
        ['story_name', 'page_num']
    ).cumcount()
    stim_df = stim_df.drop(columns=['_row_order'], errors='ignore')

    unexpected_stories = set(stim_df['story_name'].unique()) - ROAMM_STORY_NAMES
    if unexpected_stories:
        logger.warning(f'Unexpected ROAMM stimulus story_name values: {unexpected_stories}')

    # Every downstream merge assumes fix_fixed_word_key is unique per
    # (story_name, page_num) - i.e. one row per word occurrence. If the
    # wiki_stories folder has duplicate/overlapping files (e.g. both a
    # "_coordinates.csv" and a "_control_coordinates.csv" for the same story,
    # or genuinely duplicated rows), this key stops being unique and every
    # merge keyed on it silently multiplies rows instead of erroring - the
    # likely cause of a combinatorial memory blowup. Fail loudly here instead.
    dup_mask = stim_df.duplicated(
        subset=['story_name', 'page_num', 'fix_fixed_word_key'], keep=False
    )
    if dup_mask.any():
        n_dup = dup_mask.sum()
        example = stim_df.loc[dup_mask, ['story_name', 'page_num', 'fix_fixed_word_key']].head(3)
        raise ValueError(
            f'{n_dup} duplicate (story_name, page_num, fix_fixed_word_key) rows found '
            f'across the wiki_stories CSVs - this key must be unique per word occurrence '
            f'or downstream merges will multiply combinatorially instead of erroring. '
            f'Check for duplicate/overlapping files in wiki_stories/ (e.g. both '
            f'"pluto_coordinates.csv" and "pluto_control_coordinates.csv" present at once). '
            f'Example duplicated rows:\n{example}'
        )

    logger.info(f'Loaded {len(stim_df)} stimulus word rows across {stim_df["story_name"].nunique()} stories.')

    return stim_df


def _roamm_fixation_report_to_ia_report(
    fix_report: pd.DataFrame, stim_df: pd.DataFrame
) -> pd.DataFrame:
    """One row per (participant, story, page, word) - including words the
    participant never fixated, so a real IA_SKIP / skip rate is possible."""
    fix_word_keys = ['participant_id', 'story_name', 'page_num', 'fix_fixed_word_key']
    fix_agg = (
        fix_report.sort_values(
            ['participant_id', 'story_name', 'page_num', 'CURRENT_FIX_INDEX']
        )
        .groupby(fix_word_keys, as_index=False)
        .agg(
            IA_DWELL_TIME=('CURRENT_FIX_DURATION', 'sum'),
            IA_FIXATION_COUNT=('CURRENT_FIX_DURATION', 'count'),
            IA_FIRST_FIXATION_DURATION=('CURRENT_FIX_DURATION', 'first'),
        )
    )

    # cross-join every participant/run against the full per-page stimulus word
    # list, then left-merge in what was actually fixated
    participants = fix_report.groupby(
        ['participant_id', 'story_name', 'page_num', 'run_num'], as_index=False
    ).agg(is_mw=('is_mw', 'max'), page_dur=('page_dur', 'first'))

    # this cross-join is intentionally one-to-many (each participant/page
    # matched against every word on that page) - but it must be exactly
    # one-to-many, not many-to-many. If participants has duplicate rows per
    # (participant, story, page) - e.g. from inconsistent run_num values in
    # messy raw data - the expansion below multiplies further than intended,
    # which is the likely source of an unbounded memory blowup. Guard it.
    dup_participants = participants.duplicated(
        subset=['participant_id', 'story_name', 'page_num'], keep=False
    )
    if dup_participants.any():
        example = participants.loc[
            dup_participants, ['participant_id', 'story_name', 'page_num', 'run_num']
        ].head(5)
        raise ValueError(
            f'{dup_participants.sum()} duplicate (participant_id, story_name, page_num) rows '
            f'in participants - likely inconsistent run_num values for the same page in the '
            f'raw data. This would multiply the word-list cross-join below far beyond the '
            f'intended size. Example duplicated rows:\n{example}'
        )

    expected_size = sum(
        len(participants[
            (participants['story_name'] == story) & (participants['page_num'] == page)
        ])
        * len(stim_df[(stim_df['story_name'] == story) & (stim_df['page_num'] == page)])
        for story, page in participants[['story_name', 'page_num']].drop_duplicates().itertuples(index=False)
    )
    logger.info(
        f'Cross-joining {len(participants)} participant-pages against stimulus words - '
        f'expected result size: {expected_size} rows.'
    )

    full_grid = participants.merge(
        stim_df[
            [
                'story_name', 'page_num', 'fix_fixed_word_key', 'IA_LABEL',
                'word_index_in_text', 'IA_LEFT', 'IA_TOP', 'IA_WIDTH', 'IA_HEIGHT',
            ]
        ],
        on=['story_name', 'page_num'],
        how='left',
    )
    if len(full_grid) != expected_size:
        raise ValueError(
            f'full_grid has {len(full_grid)} rows but expected {expected_size} - '
            f'the story_name/page_num join key is producing an unexpected cardinality. '
            f'Stopping before this multiplies further downstream.'
        )

    ia = full_grid.merge(
        fix_agg,
        on=['participant_id', 'story_name', 'page_num', 'fix_fixed_word_key'],
        how='left',
        validate='many_to_one',
    )
    ia['IA_FIXATION_COUNT'] = ia['IA_FIXATION_COUNT'].fillna(0)
    ia['IA_DWELL_TIME'] = ia['IA_DWELL_TIME'].fillna(0)
    ia['IA_SKIP'] = (ia['IA_FIXATION_COUNT'] == 0).astype(int)
    return ia


def _attach_next_saccade_features(fix_report: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Attach the outgoing saccade following each fixation (NEXT_SAC_DURATION,
    NEXT_SAC_AMPLITUDE, NEXT_SAC_AVG_VELOCITY), required by
    compute_fixation_trial_level_features. Uses merge_asof to match each
    fixation's end time to the nearest saccade starting at or after it,
    within the same trial."""
    saccades = samples[samples['is_saccade'] == True]  # noqa: E712
    sacc_agg = saccades.groupby(
        ['participant_id', 'story_name', 'page_num', 'sacc_tStart', 'sacc_tEnd'],
        as_index=False,
    ).agg(
        NEXT_SAC_DURATION=('sacc_duration', 'first'),
        NEXT_SAC_AMPLITUDE=('sacc_ampDeg', 'first'),
        NEXT_SAC_AVG_VELOCITY=('sacc_vPeak', 'first'),
    )

    fix_sorted = fix_report.sort_values('fix_tEnd').reset_index(drop=True)
    sacc_sorted = sacc_agg.sort_values('sacc_tStart').reset_index(drop=True)

    merged = pd.merge_asof(
        fix_sorted,
        sacc_sorted,
        left_on='fix_tEnd',
        right_on='sacc_tStart',
        by=['participant_id', 'story_name', 'page_num'],
        direction='forward',
    )
    return merged


def _compute_run_and_gopast_features(fix_report: pd.DataFrame) -> pd.DataFrame:
    """Per (participant, story, page, word_index_in_text): IA_FIRST_RUN_DWELL_TIME
    (total duration of the first consecutive run of fixations on that word)
    and IA_SELECTIVE_REGRESSION_PATH_DURATION / go-past time (sum of fixation
    durations from first reaching the word until first moving past it to the
    right, including any leftward regressions along the way). Definition
    follows the go-past time description already cited in utils.py
    (https://tmalsburg.github.io/MeziereEtAl2021MS.pdf).

    Implemented as a single linear pass over the whole sorted array (tracking
    trial boundaries manually), NOT a Python-level loop over
    fix_report.groupby(trial_cols) groups - the latter pays pandas per-group
    overhead (DataFrame slicing/object creation) on every iteration, which
    becomes catastrophic if the real data has far more distinct trial groups
    than expected (e.g. due to a type/precision quirk fragmenting the
    grouping key), even though the underlying algorithm is O(n) either way.
    """
    trial_cols = ['participant_id', 'story_name', 'page_num']
    fix_sorted = fix_report.sort_values(trial_cols + ['CURRENT_FIX_INDEX']).reset_index(drop=True)
    n = len(fix_sorted)

    if fix_sorted['word_index_in_text'].isna().any():
        raise ValueError(
            'fix_report contains NaN word_index_in_text rows - these must be dropped '
            'before this function runs, since NaN == NaN is always False in Python and '
            'silently causes the run-detection loop below to never advance (infinite loop) '
            'rather than erroring.'
        )

    # integer trial id via vectorized boundary detection - no python loop
    trial_changed = (fix_sorted[trial_cols] != fix_sorted[trial_cols].shift()).any(axis=1)
    trial_id = trial_changed.cumsum().to_numpy() - 1

    boundary_starts = np.concatenate([[0], np.where(np.diff(trial_id) != 0)[0] + 1, [n]])
    # trial_end_for_row[i] = exclusive end index of the trial containing row i
    trial_of_row = np.searchsorted(boundary_starts, np.arange(n), side='right') - 1
    trial_end_for_row = boundary_starts[trial_of_row + 1]

    # first row of each trial, used to map trial_id back to its (subject, page) key
    trial_key_rows = fix_sorted[trial_cols].iloc[boundary_starts[:-1]].reset_index(drop=True)

    ia_ids = fix_sorted['word_index_in_text'].tolist()
    durations = fix_sorted['CURRENT_FIX_DURATION'].tolist()

    # --- first-run dwell time: single linear pass, two-pointer per run ---
    first_run_dwell: dict = {}
    first_reach_idx: dict = {}
    i = 0
    while i < n:
        t = trial_id[i]
        word = ia_ids[i]
        j = i
        run_total = 0.0
        while j < n and trial_id[j] == t and ia_ids[j] == word:
            run_total += durations[j]
            j += 1
        key = (t, word)
        if key not in first_run_dwell:
            first_run_dwell[key] = run_total
            first_reach_idx[key] = i
        i = j

    # --- go-past time: monotonic stack over the whole array, reset at trial
    # boundaries so nothing leaks across trials. Still O(n) total: each index
    # is pushed and popped at most once across the entire pass. ---
    next_greater = [0] * n
    stack: list[int] = []
    current_trial = -1
    for i in range(n):
        if trial_id[i] != current_trial:
            while stack:
                idx = stack.pop()
                next_greater[idx] = trial_end_for_row[idx]
            current_trial = trial_id[i]
        while stack and ia_ids[stack[-1]] < ia_ids[i]:
            next_greater[stack.pop()] = i
        stack.append(i)
    while stack:
        idx = stack.pop()
        next_greater[idx] = trial_end_for_row[idx]

    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + durations[i]

    go_past: dict = {}
    for key, start_i in first_reach_idx.items():
        stop = next_greater[start_i]
        go_past[key] = prefix[stop] - prefix[start_i]

    records = []
    for (t, word), dwell in first_run_dwell.items():
        key_row = trial_key_rows.iloc[t]
        records.append(
            {
                'participant_id': key_row['participant_id'],
                'story_name': key_row['story_name'],
                'page_num': key_row['page_num'],
                'word_index_in_text': word,
                'IA_FIRST_RUN_DWELL_TIME': dwell,
                'IA_SELECTIVE_REGRESSION_PATH_DURATION': go_past.get((t, word), dwell),
            }
        )

    return pd.DataFrame.from_records(records)


def combine_roamm(data_path: Path) -> None:
    data_args = get_data_args('ROAMM')
    base = data_args.base_path

    samples = _load_all_roamm_runs(base / 'subject_ml_data')
    logger.info(f'Loaded {len(samples)} sample rows.')
    stim_df = _load_roamm_stimuli(base / 'wiki_stories')

    fix_report = _roamm_samples_to_fixation_report(samples)
    logger.info(f'Built fixation report: {len(fix_report)} rows.')

    # CURRENT_FIX_INTEREST_AREA_INDEX (standardized from this) needs a
    # numerically-ordered word index for regression/progression detection in
    # add_missing_features (it does < / > comparisons). fix_fixed_word_key is
    # a UUID string, not orderable, so attach the positional word_index_in_text
    # from the stimulus files - the same ID space the IA report uses.
    # validate='many_to_one': each fixation should match at most one stimulus
    # word - if it doesn't, this fails immediately instead of silently
    # multiplying every fixation row (the likely cause of a combinatorial
    # memory blowup, rather than quietly hanging for an hour before OOMing).
    n_before = len(fix_report)
    fix_report = fix_report.merge(
        stim_df[['story_name', 'page_num', 'fix_fixed_word_key', 'word_index_in_text']],
        on=['story_name', 'page_num', 'fix_fixed_word_key'],
        how='left',
        validate='many_to_one',
    )
    assert len(fix_report) == n_before, (
        f'fix_report row count changed from {n_before} to {len(fix_report)} after merging '
        f'word_index_in_text - this merge should be many-to-one and row-count-preserving.'
    )

    # Fixations whose fix_fixed_word_key didn't match any stimulus word end up
    # with word_index_in_text = NaN. These can't contribute to word-level
    # features and, left in, cause an infinite loop in
    # _compute_run_and_gopast_features (NaN == NaN is always False, so its
    # two-pointer run-detection loop never advances past a NaN row). Drop them
    # here and log how many, since a non-trivial count would indicate a real
    # upstream data-matching problem worth investigating separately.
    n_unmatched = fix_report['word_index_in_text'].isna().sum()
    if n_unmatched:
        logger.warning(
            f'{n_unmatched} fixation(s) ({n_unmatched / len(fix_report):.2%}) had no matching '
            f'stimulus word (fix_fixed_word_key not found in wiki_stories) and are being '
            f'dropped. If this is a large fraction, investigate the word_key matching '
            f'between subject_ml_data and wiki_stories rather than trusting this silently.'
        )
        fix_report = fix_report.dropna(subset=['word_index_in_text']).copy()

    fix_report = _attach_next_saccade_features(fix_report, samples)
    logger.info(f'Attached saccade features: {len(fix_report)} rows (should be unchanged).')

    ia_report = _roamm_fixation_report_to_ia_report(fix_report, stim_df)
    logger.info(f'Built IA report: {len(ia_report)} rows.')

    run_gopast = _compute_run_and_gopast_features(fix_report)
    n_before = len(ia_report)
    ia_report = ia_report.merge(
        run_gopast,
        on=['participant_id', 'story_name', 'page_num', 'word_index_in_text'],
        how='left',
        validate='many_to_one',
    )
    assert len(ia_report) == n_before, (
        f'ia_report row count changed from {n_before} to {len(ia_report)} after merging '
        f'run/go-past features - this merge should be many-to-one and row-count-preserving.'
    )
    # words never fixated have no run/go-past values by construction - 0 is the
    # correct value (no time spent), not a missing-data gap
    ia_report['IA_FIRST_RUN_DWELL_TIME'] = ia_report['IA_FIRST_RUN_DWELL_TIME'].fillna(0)
    ia_report['IA_SELECTIVE_REGRESSION_PATH_DURATION'] = ia_report[
        'IA_SELECTIVE_REGRESSION_PATH_DURATION'
    ].fillna(0)

    (base / 'precomputed_events').mkdir(parents=True, exist_ok=True)
    (base / 'precomputed_reading_measures').mkdir(parents=True, exist_ok=True)
    output_fix_csv = base / 'precomputed_events' / 'combined_fixations.csv'
    output_ia_csv = base / 'precomputed_reading_measures' / 'combined_ia.csv'
    fix_report.to_csv(output_fix_csv, index=False)
    ia_report.to_csv(output_ia_csv, index=False)
    logger.info(f'Combined CSV saved: {output_fix_csv}')
    logger.info(f'Combined CSV saved: {output_ia_csv}')

    summary_df = summarize_dataframe(fix_report, DataSets.ROAMM)
    summary_df.to_csv(STATS_FOLDER / f'{DataSets.ROAMM}_raw_fixations_summary.csv', index=False)
    summary_df = summarize_dataframe(ia_report, DataSets.ROAMM)
    summary_df.to_csv(STATS_FOLDER / f'{DataSets.ROAMM}_raw_ia_summary.csv', index=False)


def combine_dataset(dataset_name: str) -> None:
    logger.info(f'Processing {dataset_name}...')
    if dataset_name == DataSets.ROAMM:
        combine_roamm(Path('data'))

    elif dataset_name == 'MECOL2':
        lookup = {}
        for part in ('W1', 'W2'):
            lookup[f'data_args_{part}'] = get_data_args(f'{dataset_name}{part}')
            base = lookup[f'data_args_{part}'].base_path
            logger.info(f'Processing {dataset_name}{part}...')
            dataset_def = pm.DatasetLibrary.get(f'{dataset_name}{part}')
            dataset_def.resources = ResourceDefinitions(
                [resource for resource in dataset_def.resources if resource.content != 'gaze']
            )
            logger.info(f'Loading {dataset_name}{part} dataset...')
            dataset = pm.Dataset(dataset_def, f'data/{dataset_name}{part}').load()
            if part == 'W1':
                dataset.precomputed_events[0].frame = (
                    dataset.precomputed_events[0]
                    .frame.to_pandas()
                    .drop('trial', axis=1)
                )
                dataset.precomputed_reading_measures[
                    0
                ].frame = dataset.precomputed_reading_measures[0].frame.to_pandas()
            if part == 'W2':
                dataset.precomputed_events[0].frame = dataset.precomputed_events[
                    0
                ].frame.to_pandas()
                dataset.precomputed_reading_measures[0].frame = (
                    dataset.precomputed_reading_measures[0]
                    .frame.to_pandas()
                    .drop('supplementary_id', axis=1)
                )
            lookup[f'dataset_{part}'] = dataset
        fix = pd.concat(
            [
                lookup['dataset_W1'].precomputed_events[0].frame,
                lookup['dataset_W2'].precomputed_events[0].frame,
            ],
            ignore_index=True,
        )
        ia = pd.concat(
            [
                lookup['dataset_W1'].precomputed_reading_measures[0].frame,
                lookup['dataset_W2'].precomputed_reading_measures[0].frame,
            ],
            ignore_index=True,
        )
        base = Path('data/MECOL2')
        base.mkdir(parents=True, exist_ok=True)
        Path(base / 'precomputed_events').mkdir(parents=True, exist_ok=True)
        Path(base / 'precomputed_reading_measures').mkdir(parents=True, exist_ok=True)
        logger.info(f'Total combined rows precomputed events: {len(fix)}')
        output_csv = base / 'precomputed_events' / 'combined_fixations.csv'
        output_summary_csv = STATS_FOLDER / f'{dataset_name}_raw_fixations_summary.csv'
        fix.to_csv(output_csv, index=False)
        logger.info(f'Combined CSV saved: {output_csv}')
        summary_df = summarize_dataframe(fix, dataset_name)
        summary_df.to_csv(output_summary_csv, index=False)
        logger.info(f'Column summary saved: {output_summary_csv}')
        logger.info(f'Total combined rows precomputed reading measures: {len(ia)}')
        output_csv = base / 'precomputed_reading_measures' / 'combined_ia.csv'
        output_summary_csv = STATS_FOLDER / f'{dataset_name}_raw_ia_summary.csv'
        ia.to_csv(output_csv, index=False)
        logger.info(f'Combined CSV saved: {output_csv}')
        summary_df = summarize_dataframe(ia, dataset_name)
        summary_df.to_csv(output_summary_csv, index=False)
        logger.info(f'Column summary saved: {output_summary_csv}')

    else:
        data_args = get_data_args(dataset_name)
        base = data_args.base_path

        dataset_def = pm.DatasetLibrary.get(dataset_name)
        dataset_def.resources = pm.ResourceDefinitions(
            [resource for resource in dataset_def.resources if resource.content != 'gaze']
        )

        logger.info(f'Loading {dataset_name} dataset...')
        dataset = pm.Dataset(dataset_def, f'data/{dataset_name}').load()

        if dataset.definition.resources.has_content('precomputed_events'):
            logger.info('Processing precomputed events...')
            combine_files(
                dataset=dataset.precomputed_events,
                fileinfo=dataset.fileinfo['precomputed_events'],
                output_csv=base / 'precomputed_events' / 'combined_fixations.csv',
                output_summary_csv=STATS_FOLDER
                / f'{dataset_name}_raw_fixations_summary.csv',
                dataset_name=dataset_name,
            )
        else:
            logger.info(f'{dataset_name} has no precomputed events...')

        if dataset.definition.resources.has_content('precomputed_reading_measures'):
            logger.info('Processing precomputed reading measures...')
            combine_files(
                dataset=dataset.precomputed_reading_measures,
                fileinfo=dataset.fileinfo['precomputed_reading_measures'],
                output_csv=base / 'precomputed_reading_measures' / 'combined_ia.csv',
                output_summary_csv=STATS_FOLDER / f'{dataset_name}_raw_ia_summary.csv',
                dataset_name=dataset_name,
            )
        else:
            logger.info(f'{dataset_name} has no precomputed reading measures...')

        if dataset_name == 'SBSAT':
            if data_args.raw_ia_dir:
                logger.info('Combining stimulus files...')
                combine_stimulus_files(
                    data_path=data_args.raw_ia_dir,
                    matching_pattern='*words.csv',
                    dataset_name=dataset_name,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='')
    args = parser.parse_args()

    if args.dataset:
        datasets = args.dataset.split(',')
    else:
        datasets = DataSets

    for dataset_name in datasets:
        combine_dataset(dataset_name)


if __name__ == '__main__':
    raise SystemExit(main())
