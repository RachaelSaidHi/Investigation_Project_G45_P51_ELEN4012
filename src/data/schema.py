"""Common manifest schema shared by all datasets.

MANIFEST_COLUMNS defines the columns, and their order, of every manifest
(full, study, analysis and dev). REQUIRED_COLUMNS must be filled for every
recording. Other columns may be blank where a dataset does not provide the
information, or until the stage that fills them (for example, accent_group
at study selection and processed_filepath at analysis selection).
"""

MANIFEST_COLUMNS = [
    "sample_id",
    "dataset",
    "source_split",
    "source_sample_id",
    "filepath",
    "processed_filepath",
    "speaker_id",
    "accent_raw",
    "accent_group",
    "country",
    "native_language",
    "birthplace",
    "sex",
    "age",
    "age_group",
    "age_onset",
    "duration_s",
    "original_sample_rate",
    "domain",
    "reference_transcript",
]

REQUIRED_COLUMNS = [
    "sample_id",
    "dataset",
    "source_sample_id",
    "filepath",
    "speaker_id",
    "accent_raw",
    "duration_s",
    "original_sample_rate",
    "reference_transcript",
]