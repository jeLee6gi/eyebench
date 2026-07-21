from __future__ import annotations

import pandas as pd
import spacy
from loguru import logger
from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors.extractor_switch import get_surp_extractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType

from src.configs.constants import DataType, Fields
from src.data.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.data.utils import (
    add_missing_features,
    compute_trial_level_features,
    replace_missing_values,
)

logger.add('logs/preprocessing.log', level='INFO')


class ROAMMProcessor(DatasetProcessor):
    """
    Processor for the ROAMM (Reading Observed At Mindless Moments) dataset.

    ROAMM is distributed as sample-level EEG + eye-tracking pandas DataFrames
    rather than EyeLink-style Interest Area / Fixation reports. union_raw_files.py
    (combine_roamm) converts the raw per-subject .pkl files and the wiki_stories
    stimulus/AOI files into combined_ia.csv / combined_fixations.csv, already in
    the shape every other EyeBench processor expects (one row per word, one row
    per fixation, including unfixated stimulus words with IA_FIXATION_COUNT=0).
    This class standardizes column names, attaches linguistic features (POS,
    surprisal, dependency parse) computed once per page from the true stimulus
    text, and derives the two ROAMM task labels: RC (page-level comprehension
    question correctness) and MW (page-level mind-wandering detection).
    """

    IA_COLUMN_MAP = {
        'participant_id': Fields.SUBJECT_ID,
        'page_num': Fields.UNIQUE_PARAGRAPH_ID,
        'word_index_in_text': Fields.IA_DATA_IA_ID_COL_NAME,
    }

    FIXATION_COLUMN_MAP = {
        'participant_id': Fields.SUBJECT_ID,
        'page_num': Fields.UNIQUE_PARAGRAPH_ID,
        'word_index_in_text': Fields.FIXATION_REPORT_IA_ID_COL_NAME,
    }

    LINGUISTIC_FEATURE_RENAMES = {
        'POS': 'universal_pos',
        'Reduced_POS': 'ptb_pos',
        'Entity': 'entity_type',
        'Is_Content_Word': 'is_content_word',
        'Length': 'word_length_no_punctuation',
        'n_Lefts': 'left_dependents_count',
        'n_Rights': 'right_dependents_count',
        'Distance2Head': 'distance_to_head',
        'Head_word_idx': 'head_word_index',
        'Dependency_Relation': 'dependency_relation',
        'Head_Direction': 'head_direction',
        'gpt2_Surprisal': 'gpt2_surprisal',
    }

    # reading_metadata.csv's `reading` column holds human-readable story titles,
    # not a page index - map them to the snake_case story_name used everywhere
    # else in this pipeline (derived from the wiki_stories/subject_ml_data files).
    STORY_TITLE_TO_NAME = {
        'Pluto': 'pluto',
        'Prisoners Dilemma': 'prisoners_dilemma',
        'History of Film': 'history_of_film',
        'Serena Williams': 'serena_williams',
        'The Voynich Manuscript': 'the_voynich_manuscript',
    }

    def get_column_map(self, data_type: DataType) -> dict[str, str]:
        return self.IA_COLUMN_MAP if data_type == DataType.IA else self.FIXATION_COLUMN_MAP

    def get_columns_to_keep(self) -> list[str]:
        return []

    def _add_unique_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # standardize_column_names already renamed the raw 'page_num' column to
        # Fields.UNIQUE_PARAGRAPH_ID before dataset_specific_processing runs, so
        # at this point df[UNIQUE_PARAGRAPH_ID] still holds the bare page number
        # (not yet combined with story_name). Stash it under its own name first,
        # since _load_and_merge_labels needs to join reading_metadata.csv on the
        # raw page number, and this column gets overwritten on the very next line.
        df['page_num'] = df[Fields.UNIQUE_PARAGRAPH_ID]
        df[Fields.UNIQUE_PARAGRAPH_ID] = (
            df['story_name'].astype(str) + '_' + df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )
        df[Fields.UNIQUE_TRIAL_ID] = (
            df[Fields.SUBJECT_ID].astype(str) + '_' + df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )
        return df

    @staticmethod
    def _normalize_participant_id(series: pd.Series) -> pd.Series:
        """Normalize participant IDs to the 's<digits>' string form used
        throughout this pipeline (derived from subject_ml_data folder names).
        Some label files store bare integers (10014) instead of 's10014'."""
        return series.apply(
            lambda value: f's{int(value)}' if str(value).strip().isdigit() else str(value).strip()
        )

    @staticmethod
    def _parse_int_array_string(value: str) -> list[int]:
        """Parse reading_metadata.csv's stringified numpy arrays, e.g.
        '[1 0 0 1 1 1 0 1 0 0]' -> [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]."""
        return [int(float(token)) for token in str(value).strip('[]').split()]

    def _load_and_merge_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # participant demographics (age, gender, handedness, ADHD, etc.)
        demo_df = pd.read_csv('data/ROAMM/subject_demographics_information.csv').rename(
            columns={
                'participant_id': Fields.SUBJECT_ID,
                'adhd?_y/n': 'adhd',
                'reading_disability?_y/n': 'reading_disability',
            }
        )
        demo_df[Fields.SUBJECT_ID] = self._normalize_participant_id(demo_df[Fields.SUBJECT_ID])
        # keep the *_clean columns as the canonical gender/handedness fields,
        # drop the raw free-text ones to avoid duplicate/ambiguous columns
        demo_df = demo_df.drop(columns=['gender', 'handness'], errors='ignore').rename(
            columns={'gender_clean': 'gender', 'handness_clean': 'handedness'}
        )
        df = df.merge(demo_df, on=Fields.SUBJECT_ID, how='left', validate='many_to_one')

        # reading_metadata.csv has one row per (participant, story) - not per
        # page. `reading` holds the story's human-readable title (e.g.
        # "Prisoners Dilemma"), and `is_correct` is a stringified array of 10
        # per-page comprehension outcomes for that story, e.g.
        # "[1 0 0 1 1 1 0 1 0 0]". We map the title to our snake_case
        # story_name, explode the array into one row per page (0-indexed by
        # array position, matching page ordering everywhere else in this
        # pipeline), and join on (participant_id, story_name, page_num) -
        # sidestepping any need to align `run` with our run_num at all.
        cq_df = pd.read_csv('data/ROAMM/reading_metadata.csv').rename(
            columns={'sub_id': Fields.SUBJECT_ID}
        )
        cq_df[Fields.SUBJECT_ID] = self._normalize_participant_id(cq_df[Fields.SUBJECT_ID])

        unmapped_titles = set(cq_df['reading'].unique()) - set(self.STORY_TITLE_TO_NAME)
        if unmapped_titles:
            raise ValueError(
                f'Unrecognized story titles in reading_metadata.csv: {unmapped_titles}. '
                f'STORY_TITLE_TO_NAME needs updating to cover these.'
            )
        cq_df['story_name'] = cq_df['reading'].map(self.STORY_TITLE_TO_NAME)

        cq_df['is_correct'] = cq_df['is_correct'].apply(self._parse_int_array_string)
        cq_long = cq_df.explode('is_correct', ignore_index=True)
        cq_long['page_num'] = cq_long.groupby(
            [Fields.SUBJECT_ID, 'story_name']
        ).cumcount()
        cq_long['is_correct'] = cq_long['is_correct'].astype(int)
        cq_long = cq_long[[Fields.SUBJECT_ID, 'story_name', 'page_num', 'is_correct']]

        common_subjects = set(df[Fields.SUBJECT_ID]) & set(cq_long[Fields.SUBJECT_ID])
        if not common_subjects:
            raise ValueError(
                'No overlapping participant IDs between reading_metadata.csv and the '
                'fixation/IA data - check that sub_id / participant_id use the same '
                'subject ID format (e.g. both "s10014" vs one being bare "10014").'
            )

        df['page_num'] = pd.to_numeric(df['page_num'], errors='coerce').astype('Int64')
        cq_long['page_num'] = pd.to_numeric(cq_long['page_num'], errors='coerce').astype('Int64')

        df = df.merge(
            cq_long,
            on=[Fields.SUBJECT_ID, 'story_name', 'page_num'],
            how='left',
            validate='many_to_one',
        )
        match_rate = df['is_correct'].notna().groupby(df[Fields.UNIQUE_TRIAL_ID]).max().mean()
        logger.info(
            f'reading_metadata.csv matched a comprehension-question outcome for '
            f'{match_rate:.1%} of trials via (participant_id, story_name, page_num). '
            f'If this is unexpectedly low, the per-page array-position assumption '
            f'(array index == page_num) is likely wrong and needs to be revisited.'
        )

        # Nothing downstream (add_missing_features / compute_trial_level_features /
        # base_dataset.py) drops rows with a null target_column - PoTeCProcessor
        # establishes the precedent of dropping trials lacking a comprehension
        # outcome upfront instead, which we follow here. This also removes those
        # trials from the MW dataset, matching how PoTeC's dropna affects both
        # PoTeC_DE and PoTeC_RC even though only one of them needs it.
        df = df.dropna(subset=['is_correct']).copy()

        # RC: page-level MCQ correctness (binary, matches EyeBench's existing RC task)
        df['RC'] = df['is_correct'].astype(int)

        # MW: was any mind-wandering episode reported on this page? (binary)
        # Some trials have is_mw as NaN across every fixation on that page
        # (no annotation at all, not just a missing single sample), so max()
        # over an all-NaN group stays NaN. Defaulting that to 0 ("not
        # mind-wandering") is a modeling assumption, not a data-cleaning
        # no-brainer - surfacing the affected count rather than silently
        # deciding. Revisit if this count is large relative to total trials.
        mw_raw = df.groupby(Fields.UNIQUE_TRIAL_ID)['is_mw'].transform('max')
        mw_nan_count = mw_raw.isna().sum()
        if mw_nan_count:
            logger.warning(
                f'{mw_nan_count} row(s) ({mw_nan_count / len(df):.1%} of the IA table) '
                f'have no is_mw annotation at all for their page and are defaulting to '
                f'MW=0 (not mind-wandering). Confirm this default is appropriate.'
            )
        df['MW'] = mw_raw.fillna(0).astype(int)

        # PARAGRAPH_RT: needed by calc_reading_speed (compute_ia_trial_level_features).
        # page_dur is already propagated onto both IA and FIXATIONS by combine_roamm.
        df['PARAGRAPH_RT'] = df['page_dur']

        return df

    def _add_linguistic_features(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        """Compute POS/surprisal/dependency features once per unique page
        (ROAMM's 5 articles x 10 pages are fixed stimuli shared across all
        participants), then merge onto every stimulus word (fixated or not).
        standardize_column_names already renamed 'word_index_in_text' to
        Fields.IA_DATA_IA_ID_COL_NAME ('IA_ID') before dataset_specific_processing
        runs, so we use that name throughout rather than the pre-rename one."""
        ia_id_col = Fields.IA_DATA_IA_ID_COL_NAME
        nlp = spacy.load('en_core_web_sm')
        surp_extractor = get_surp_extractor(extractor_type=SurpExtractorType.CAT_CTX_LEFT, model_name='gpt2')

        page_words = ia_df[
            ['story_name', 'unique_paragraph_id', ia_id_col, 'IA_LABEL']
        ].drop_duplicates()

        metrics_list = []
        for (story, page), group in page_words.groupby(['story_name', 'unique_paragraph_id']):
            group = group.sort_values(ia_id_col)
            sentence = ' '.join(group['IA_LABEL'].fillna('Null').astype(str))

            metrics = get_metrics(
                target_text=sentence,
                parsing_model=nlp,
                surp_extractor=surp_extractor,
                parsing_mode='re-tokenize',
                add_parsing_features=True,
                language='en',
            )
            metrics['story_name'] = story
            metrics['unique_paragraph_id'] = page
            metrics[ia_id_col] = metrics['Token_idx']
            metrics_list.append(metrics)

        metrics_df = pd.concat(metrics_list, ignore_index=True)
        metrics_df = metrics_df.rename(columns=self.LINGUISTIC_FEATURE_RENAMES)

        merge_keys = ['story_name', 'unique_paragraph_id', ia_id_col]
        drop_cols = (set(ia_df.columns) & set(metrics_df.columns)) - set(merge_keys)
        return ia_df.merge(
            metrics_df.drop(columns=list(drop_cols) + ['Morph'], errors='ignore'),
            on=merge_keys,
            how='left',
        )

    def add_ia_report_features_to_fixation_data(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ROAMM has no precomputed EyeLink-style reading-measure columns
        (TRC_in/TRC_out, NEXT_FIX_INTEREST_AREA_INDEX, etc.) like MECO/PoTeC/
        SBSAT do - we derive them directly from the fixation sequence, then
        merge word-level IA features onto the fixation rows so both tables
        satisfy add_missing_features's column requirements. CURRENT_FIX_
        INTEREST_AREA_INDEX now holds the same positional word_index_in_text
        as the IA report's IA_ID (fixed at the union_raw_files.py level -
        it used to be the raw UUID word key, which isn't numerically orderable
        for regression detection)."""
        fix_ia_col = Fields.FIXATION_REPORT_IA_ID_COL_NAME
        ia_id_col = Fields.IA_DATA_IA_ID_COL_NAME

        fix_df = fix_df.sort_values(
            [Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, 'CURRENT_FIX_INDEX']
        ).reset_index(drop=True)

        fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] = fix_df.groupby(Fields.UNIQUE_TRIAL_ID)[
            fix_ia_col
        ].shift(-1)
        # last fixation in a trial has no "next" - fill with its own index so
        # it registers as neither a regression nor a progression
        fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] = fix_df[
            'NEXT_FIX_INTEREST_AREA_INDEX'
        ].fillna(fix_df[fix_ia_col])

        prev_fix_ia = fix_df.groupby(Fields.UNIQUE_TRIAL_ID)[fix_ia_col].shift(1)
        is_outgoing_regression = fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] < fix_df[fix_ia_col]
        is_incoming_regression = fix_df[fix_ia_col] < prev_fix_ia

        ia_keys = [Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, fix_ia_col]
        reg_out = (
            fix_df.assign(_reg_out=is_outgoing_regression)
            .groupby(ia_keys)['_reg_out']
            .sum()
            .rename('IA_REGRESSION_OUT_FULL_COUNT')
        )
        reg_in = (
            fix_df.assign(_reg_in=is_incoming_regression)
            .groupby(ia_keys)['_reg_in']
            .sum()
            .rename('IA_REGRESSION_IN_COUNT')
        )

        ia_df = ia_df.merge(
            reg_out, left_on=[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, ia_id_col],
            right_index=True, how='left',
        )
        ia_df = ia_df.merge(
            reg_in, left_on=[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, ia_id_col],
            right_index=True, how='left',
        )
        ia_df['IA_REGRESSION_OUT_FULL_COUNT'] = ia_df['IA_REGRESSION_OUT_FULL_COUNT'].fillna(0)
        ia_df['IA_REGRESSION_IN_COUNT'] = ia_df['IA_REGRESSION_IN_COUNT'].fillna(0)

        # merge word-level IA features onto the fixation report - both tables
        # now share the same positional word-index space (see docstring above)
        ia_feature_cols = [
            'IA_LABEL', 'word_length', 'word_length_no_punctuation', 'universal_pos',
            'ptb_pos', 'entity_type', 'is_content_word', 'gpt2_surprisal',
            'left_dependents_count', 'right_dependents_count', 'distance_to_head',
            'head_word_index', 'dependency_relation', 'head_direction',
            'IA_LEFT', 'IA_TOP', 'IA_WIDTH', 'IA_HEIGHT',
            'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_FULL_COUNT', 'IA_FIXATION_COUNT',
            'IA_DWELL_TIME', 'IA_FIRST_FIXATION_DURATION', 'IA_SKIP',
        ]
        ia_feature_cols = [c for c in ia_feature_cols if c in ia_df.columns]
        merge_keys = [Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, ia_id_col]
        fix_df = fix_df.merge(
            ia_df[merge_keys + ia_feature_cols].rename(columns={ia_id_col: fix_ia_col}).drop_duplicates(
                subset=[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, fix_ia_col]
            ),
            on=merge_keys[:-1] + [fix_ia_col],
            how='left',
            validate='many_to_one',
        )

        return ia_df, fix_df

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type not in data_dict or data_dict[data_type] is None:
                continue

            df = data_dict[data_type]
            df = self._add_unique_ids(df)
            df = self._load_and_merge_labels(df)
            data_dict[data_type] = df

        # linguistic features are computed on the IA (word-level) report, which
        # already contains every stimulus word - including ones nobody fixated -
        # via combine_roamm's cross-join against the wiki_stories stimulus files.
        data_dict[DataType.IA] = self._add_linguistic_features(data_dict[DataType.IA])

        # TextDataSet requires a 'paragraph' column (full page text) on the IA
        # table - mirrors SBSATProcessor's precedent exactly. Must sort by word
        # order first, since groupby().transform(' '.join) uses row order as-is.
        # Note: unlike SBSAT/PoTeC, ROAMM has no 'question' column here - the
        # real MCQ wording isn't recoverable from reading_metadata.csv (only
        # numeric response/answer-option indices are available), so ROAMM_RC
        # will train on paragraph text only, without literal question-text
        # conditioning. TextDataSet degrades gracefully when 'question' is
        # absent rather than erroring, but this is a real modeling-relevant
        # gap worth being aware of, not just a technical footnote.
        data_dict[DataType.IA] = data_dict[DataType.IA].sort_values(
            [Fields.UNIQUE_TRIAL_ID, Fields.IA_DATA_IA_ID_COL_NAME]
        )
        data_dict[DataType.IA]['paragraph'] = data_dict[DataType.IA].groupby(
            Fields.UNIQUE_TRIAL_ID
        )['IA_LABEL'].transform(lambda x: ' '.join(x.astype(str)))

        # add_missing_features (shared util) expects a literal 'word_length'
        # column (raw character count, including punctuation) - distinct from
        # 'word_length_no_punctuation' sourced from get_metrics() above. Matches
        # MECOProcessor's precedent: ia_df['IA_LABEL'].str.len().
        data_dict[DataType.IA]['word_length'] = (
            data_dict[DataType.IA]['IA_LABEL'].astype(str).str.len()
        )

        # total_skip: compute_ia_trial_level_features needs this literal name
        # (IA_SKIP already carries the same 0/1 meaning, computed in combine_roamm).
        data_dict[DataType.IA]['total_skip'] = data_dict[DataType.IA]['IA_SKIP']

        # TRIAL_IA_COUNT: total word count per trial, needed on both tables
        # (calc_reading_speed uses it via len(trial) comparison, and the
        # BEYELSTM fixation-level features read it directly).
        trial_word_counts = data_dict[DataType.IA].groupby(Fields.UNIQUE_TRIAL_ID)[
            Fields.IA_DATA_IA_ID_COL_NAME
        ].transform('nunique')
        data_dict[DataType.IA]['TRIAL_IA_COUNT'] = trial_word_counts
        counts_by_trial = data_dict[DataType.IA].groupby(Fields.UNIQUE_TRIAL_ID)[
            Fields.IA_DATA_IA_ID_COL_NAME
        ].nunique()
        data_dict[DataType.FIXATIONS]['TRIAL_IA_COUNT'] = data_dict[DataType.FIXATIONS][
            Fields.UNIQUE_TRIAL_ID
        ].map(counts_by_trial)

        # normalized_ID: PoTeC's own precedent is a plain alias for
        # CURRENT_FIX_INTEREST_AREA_INDEX, not an actual [0,1] normalization
        # despite the name.
        data_dict[DataType.FIXATIONS]['normalized_ID'] = data_dict[DataType.FIXATIONS][
            Fields.FIXATION_REPORT_IA_ID_COL_NAME
        ]

        data_dict[DataType.IA], data_dict[DataType.FIXATIONS] = (
            self.add_ia_report_features_to_fixation_data(
                data_dict[DataType.IA], data_dict[DataType.FIXATIONS]
            )
        )

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                trial_groupby_columns=self.data_args.groupby_columns,
                mode=data_type,
            )

        for data_type in data_dict.keys():
            df = data_dict[data_type]
            data_dict[data_type] = df.loc[:, ~df.columns.duplicated()]

        trial_level_features = compute_trial_level_features(
            raw_fixation_data=data_dict[DataType.FIXATIONS],
            raw_ia_data=data_dict[DataType.IA],
            trial_groupby_columns=self.data_args.groupby_columns,
            processed_data_path=self.data_args.processed_data_path,
        )
        data_dict[DataType.TRIAL_LEVEL] = trial_level_features

        data_dict = replace_missing_values(data_dict)

        return data_dict
