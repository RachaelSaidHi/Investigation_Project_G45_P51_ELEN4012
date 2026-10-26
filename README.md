# Investigation_Project_G45_P51_ELEN4012

This repository contains the code for the ELEN4012 investigation project (Group 45, Project 51): an investigation into whether the way a speaker's accent differs acoustically from a standard baseline predicts how well a pre-trained ASR model (OpenAI Whisper) transcribes them.

## Research question

> To what extent do accent-dependent variations in acoustic speech features across African-accented English relate to transcription performance, measured using WER, in a pre-trained ASR model compared to a standard baseline?

## Overview

**Data.** Two corpora of English speech are used:

| Corpus | Used for | Notes |
|---|---|---|
| Speech Accent Archive (SAA) | American English (the baseline), Hindi (a non-African comparison group) and five African first languages: Amharic, Hausa, Kiswahili, Tigrigna and Yoruba | Every speaker reads the same short passage, so speakers are directly comparable |
| AfriSpeech-200 | Hausa, Yoruba and Kiswahili (test splits) | Different, often medical, text per speaker and many recording devices, so it is used as a second corpus to see whether the SAA results replicate, not as a direct comparison |

After quality control, the analysis manifest holds 580 recordings (407 SAA and 173 AfriSpeech). A few recordings are removed in the analysis notebook (broken transcriptions and implausible formant tracks), which leaves 535.

**Method.**
1. **Preprocessing:** recordings are converted to 16 kHz mono and put through quality control.
2. **ASR:** every recording is transcribed with Whisper (`whisper-base` is the primary model; `tiny`, `small` and `medium` are run to justify that choice) and scored with WER after text normalisation.
3. **Acoustic features:** for each recording the DSP pipeline extracts MFCCs, pitch (F0), formants (F1–F3) and speech rate, all measured over speech-active frames only.
4. **Analysis:** each speaker's features are converted to z-scores against American speakers of the same sex. Stage 1 finds which features are actually accent-dependent, without looking at WER. Stage 2 asks whether a speaker's distance from the American baseline on those features is related to their WER. Speech rate is examined separately to see how much of the WER it accounts for.

WER is the word error rate, (substitutions + deletions + insertions) / reference words.

## Project parts

The project has three parts, all of which share the same manifests and processed audio:

1. **Data and preprocessing**: acquire the Speech Accent Archive (SAA) and AfriSpeech, build manifests, convert the audio and apply quality control.
2. **Pipelines**: Whisper transcription, WER scoring, and DSP feature extraction (MFCC, pitch, formants and speech rate).
3. **Notebooks**: the choices behind the pipelines, WER by accent, and the statistical analysis of acoustic features against WER.

Results from the pipelines are joined on `sample_id`.


## Contents

- [Quick start](#quick-start-see-the-results-without-rerunning-anything)
- [1. Clone and install](#1-clone-and-install)
- [2. Set up Whisper](#2-set-up-whisper)
- [3. Acquire the Speech Accent Archive](#3-acquire-the-speech-accent-archive)
- [4. Acquire AfriSpeech](#4-acquire-afrispeech)
- [5. Build manifests and preprocess audio](#5-build-manifests-and-preprocess-audio)
- [6. What the manifests contain](#6-what-the-manifests-contain)
- [7. Running the pipelines](#7-running-the-pipelines)
- [8. Notebooks](#8-notebooks)
- [9. Testing](#9-testing)
- [10. Project repo structure](#10-project-repo-structure)
- [11. Using the shared audio loader and speech mask](#11-using-the-shared-audio-loader-and-speech-mask)
- [12. Reproducibility and limitations](#12-reproducibility-and-limitations)
- [13. Data sources, licences and citations](#13-data-sources-licences-and-citations)
- [14. Troubleshooting](#14-troubleshooting)


## Quick start: see the results without rerunning anything

The results of the full experiment are already in the repo under `results/` (the Whisper transcriptions and scored WER for each checkpoint, the decode-mode runs, and the DSP features), so you do **not** need to download the datasets or run any of the pipelines to look at the analysis.

1. Do **Section 1** (clone and install) only.
2. Open the notebooks in `notebooks/`, starting with `Accent_WER_Analysis.ipynb`, and run them. Section 8 describes what each one does.

Sections 2–7 are only needed to regenerate the results or to run the pipelines on new data.


## 1. Clone and install

```bash
git clone https://github.com/RachaelSaidHi/Investigation_Project_G45_P51_ELEN4012.git
cd Investigation_Project_G45_P51_ELEN4012
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

All commands in this README are run from the project root with the virtual environment activated, using `python -m` so that the `src` and `scripts` packages are found.


## 2. Set up Whisper
1. Create a Hugging Face account
2. Run:
```
hf auth login
```
3. Install PyTorch
* if you have a NVIDIA GPU:
```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```
* otherwise CPU only:
```
python -m pip install torch torchvision
```

Whisper checkpoints are downloaded from Hugging Face the first time they are used. Transcription on a GPU is much faster than on a CPU, particularly for the larger checkpoints.


## 3. Acquire the Speech Accent Archive


1. Download the dataset from Kaggle as ZIP file:
https://www.kaggle.com/datasets/rtatman/speech-accent-archive   
**Note: The dataset requires approximately 920 MB of disk space after extraction**

2. In the project repository, create the following directory structure if it does not already exist:

```
data/
└── raw/
    └── saa/
```

3. Extract the contents of the downloaded ZIP file into:
```   
data/raw/saa
```

4. After extraction, the folder structure should be:

```
data/
└── raw/
    └── saa/
        ├── speakers_all.csv
        ├── reading-passage.txt
        └── recordings/
            └── recordings/
                ├── afrikaans1.mp3
                ├── afrikaans2.mp3
                ├── ...
                └── yoruba5.mp3
```
The nested `recordings/recordings/` comes from the original Kaggle structure.  
The `data/raw` directory is in `.gitignore` and is not committed.  


## 4. Acquire AfriSpeech

AfriSpeech is acquired from the Hugging Face dataset:

```text
intronhealth/afrispeech-200
```

Run the acquisition script from the project root:

```bash
python -m scripts.data.acquire_afrispeech
```
**Note: The dataset requires approximately 2.15 GB of disk space after extraction**

The accent and split combinations that will be downloaded, and the dataset revision they are taken from, are specified in:

```text
configs/afrispeech_acquisition.json
```

The script downloads the required transcript metadata and audio and stores them under:

```text
data/raw/afrispeech/
```

Complete selections are skipped on re-runs and incomplete ones are downloaded again. Use `--force` to download everything again:

```bash
python -m scripts.data.acquire_afrispeech --force
```

The directory structure comes from the original AfriSpeech structure.  
The `data/raw` directory is in `.gitignore` and is not committed.

The current study uses the Hausa, Yoruba and Swahili test splits. The AfriSpeech swahili label is standardised to kiswahili in the study manifest.


## 5. Build manifests and preprocess audio 

Run the following scripts in order from the project root:

```bash
# Step 1: Build the full manifest from all locally available datasets
python -m scripts.data.build_full_manifest

# Step 2: Apply the study selection rules
python -m scripts.data.build_study_manifest

# Step 3: Preprocess the study recordings and calculate QC measurements
python -m scripts.preprocessing.preprocess_audio

# Step 4: Apply QC and build the final analysis manifest
python -m scripts.data.build_analysis_manifest

# Step 5: Build a small development subset from the analysis manifest
python -m scripts.data.build_dev_manifest
```

This produces the following files:

| File | Contents | Use |
|---|---|---|
| `full_manifest.csv` | All locally available SAA and AfriSpeech recordings in the common manifest format | Input to study selection |
| `study_manifest.csv` | Recordings meeting the dataset-specific metadata selection rules | Input to preprocessing and QC |
| `audio_qc_study_manifest.csv` | Audio quality measurements and QC exclusion decisions | Input to analysis selection |
| `analysis_manifest.csv` | Final QC-passing recordings, with one recording per retained AfriSpeech source ID | Experimental dataset |
| `dev_manifest.csv` | Small subset of the analysis manifest, containing up to one female- and one male-labelled recording per dataset/accent group | Pipeline development and testing |

The full manifest contains all usable data available locally, while the study manifest contains the samples that meet the metadata-based study selection criteria before audio quality control. The analysis manifest contains the samples used in the experiment.

The manifest CSVs are stored in `data/manifests/`, except for the QC table, which is stored in `data/processed/`.



### Audio preprocessing

All study recordings are converted to 16 kHz mono WAV files and stored in `data/processed/audio/`.

Existing WAV files are reused by default. Use `--overwrite` to regenerate them if the audio preprocessing method changes:

```bash
python -m scripts.preprocessing.preprocess_audio --overwrite
```

A different manifest can be processed with `--manifest`. The QC table is named after the input manifest.

### Quality control

Recordings are excluded if their duration is outside 5–60 seconds, they contain less than 0.05% spectral energy above 4 kHz, or their persistent-noise floor gap is below 25 dB. WADA-SNR is measured for diagnostic purposes but is not used for exclusion.

### Analysis selection

The analysis manifest retains all QC-passing SAA recordings and one randomly selected QC-passing recording per AfriSpeech source ID, using a fixed seed.

The development manifest contains a smaller subset of the analysis recordings for pipeline testing.


## 6. What the manifests contain

The full, study, analysis and development manifests use the same common schema. The QC table separately stores audio quality measurements and exclusion decisions. Important columns for the DSP, ASR and analysis pipelines include:

- `sample_id` — unique identifier for each audio recording.
- `dataset` — source dataset (`saa` or `afrispeech`).
- `source_split` — original dataset split where applicable.
- `filepath` — path to the original audio file relative to `data/raw/`.
- `processed_filepath` — path to the processed WAV file relative to `data/processed/`; populated in the analysis and development manifests.
- `speaker_id` — speaker or source identifier provided by the dataset, prefixed by dataset.
- `accent_raw` — original accent-related label provided by the source dataset.
- `accent_group` — standardised study accent category used for analysis.
- `reference_transcript` — reference text for WER computation.
- `sex` — source-provided male or female metadata.
- `age` — exact age where available.
- `age_group` — age range where available.
- `duration_s` — original recording duration in seconds.
- `domain` — speech domain or recording context.

SAA contains one recording per speaker. AfriSpeech may contain multiple recordings per source ID, but the analysis manifest retains only one QC-passing recording per ID. AfriSpeech source IDs have not been independently verified as unique speakers.

Both the DSP and ASR pipelines should use `analysis_manifest.csv` for the full experiment and retain `sample_id` in their outputs so that the results can be combined.

To load the processed audio for a manifest row, see [section 11](#11-using-the-shared-audio-loader-and-speech-mask).


## 7. Running the pipelines

All three pipelines take a manifest and a `--run-name`, and write everything for that run into its own folder under `results/`. They all work the same way:

- **Run folders:** each run has a `run_config.json` (settings, software versions, manifest hash) and a `manifest_snapshot.csv` (a copy of the manifest the run started with).
- **Checkpointing:** results are written to a CSV as the run goes, one row per `sample_id`, each with a `status` (`ok` or `error`). If a run is stopped, running the same command again resumes where it left off.
- **Safe resuming:** a run is only resumed if its configuration is unchanged (same manifest, model/feature version, settings and software versions). Otherwise the script stops and tells you what differs. Use a new `--run-name` for a new experiment.
- **Errors:** a failing sample is recorded with `status = error` and the run carries on. Use `--retry-errors` to retry those samples later, or `--fail-fast` to stop at the first error. Successful samples are never repeated.

Use `data/manifests/dev_manifest.csv` to try a pipeline out quickly, and `data/manifests/analysis_manifest.csv` for the real experiment.

### 7.1 Transcription

Transcribes every recording in a manifest with a Hugging Face Whisper checkpoint.

```bash
python -m scripts.asr.run_transcription \
    --manifest data/manifests/analysis_manifest.csv \
    --run-name analysis_base_long \
    --model openai/whisper-base \
    --decode-mode sequential_long_form
```

| Argument | Meaning |
|---|---|
| `--manifest` | Manifest CSV to transcribe (required) |
| `--run-name` | Unique name for the run; becomes the folder name (required) |
| `--model` | Hugging Face checkpoint, e.g. `openai/whisper-tiny`, `-base`, `-small`, `-medium` (required) |
| `--decode-mode` | `sequential_long_form` (default) handles recordings of any length; `short_form` transcribes a single 30 s window only |
| `--device` | `cuda`, `cuda:0` or `cpu`. Detected automatically if omitted |
| `--fp32` | Use FP32 instead of FP16 on a GPU. A CPU always uses FP32 |
| `--processed-root` | Folder `processed_filepath` values are relative to (default `data/processed`) |
| `--results-root` | Folder where run folders are created (default `results/asr/runs`) |
| `--duration-tolerance` | Allowed difference in seconds between the manifest and processed durations (default 0.5) |
| `--retry-errors`, `--fail-fast` | See above |

Output in `results/asr/runs/<run-name>/`:

| File | Contents |
|---|---|
| `run_config.json` | Model, decode mode, precision, hardware, software versions, manifest hash |
| `manifest_snapshot.csv` | The manifest the run was started with |
| `transcriptions.csv` | One row per recording: `status`, the raw Whisper transcript (`transcription_raw`), timing and real-time factor |

### 7.2 Scoring

Normalises the reference and Whisper transcripts and calculates WER for a finished (or partly finished) transcription run. No audio or model is needed, so this is quick.

```bash
python -m scripts.asr.run_scoring \
    --run-dir results/asr/runs/analysis_base_long
```

| Argument | Meaning |
|---|---|
| `--run-dir` | Transcription run folder containing `manifest_snapshot.csv` and `transcriptions.csv` (required) |
| `--output-name` | Name of the scored file (default `scored.csv`) |
| `--drop-punctuation-words` / `--no-drop-punctuation-words` | Remove or keep spoken punctuation words ("full stop", "comma", ...) in AfriSpeech. Removed by default |
| `--fail-fast` | Stop if normalisation or WER calculation fails for a sample |

Normalisation uses Whisper's English text normaliser, accessed through a fixed tokenizer (`openai/whisper-small`). AfriSpeech speakers say punctuation aloud, so by default these spoken punctuation words are removed from both the reference and the hypothesis; SAA is not affected. Scoring reruns overwrite the output file, so use `--output-name` to keep several scorings of the same run side by side. For example, the two files used in `punctuation_removal.ipynb`:

```bash
python -m scripts.asr.run_scoring \
    --run-dir results/asr/runs/analysis_base_long \
    --output-name scored_punc_removed.csv

python -m scripts.asr.run_scoring \
    --run-dir results/asr/runs/analysis_base_long \
    --output-name scored_punc_kept.csv \
    --no-drop-punctuation-words
```

Output (added to the run folder): `scored.csv` and `scored_config.json` (named after the scored file, e.g. `scored_punc_kept_config.json`). `scored.csv` has one row per recording with the normalised reference and hypothesis (`reference_norm`, `hypothesis_norm`), `wer`, `substitutions`, `deletions`, `insertions`, `hits` and `n_reference_words`, plus the transcription and scoring status.

### 7.3 Feature extraction (DSP)

Extracts acoustic features from the processed audio. Each feature family is a module in `src/dsp/features/` with a matching config in `configs/dsp/`. All features are calculated on the full waveform and then summarised over the speech-active frames only, using the shared speech mask (`configs/dsp/speech_mask.json`, see the last section of this README).

| Feature | Module | Summary per recording |
|---|---|---|
| `mfcc` | `mfcc.py` | Mean and SD of 13 MFCCs |
| `pitch` | `pitch.py` | F0 median, span (semitones), 10th and 90th percentile, voiced frames (Praat) |
| `formants` | `formants.py` | F1–F3 summaries (Praat Burg) using all voiced frames and the loudest half of them |
| `speech_rate_reference` | `speech_rate_reference.py` | Speaking rate, articulation rate and pause measures, using syllables counted in the **reference transcript** (not Whisper's output) |
| `speech_rate_acoustic` | `speech_rate_acoustic.py` | Transcript-free speech rate from syllable nuclei detected in the audio |
| `rms_test` | `rms_test.py` | A trivial RMS feature used as a minimal example of a feature module |

```bash
# One feature
python -m scripts.dsp.run_feature_extraction \
    --manifest data/manifests/analysis_manifest.csv \
    --run-name analysis_full \
    --feature mfcc

# Several features, one after the other
python -m scripts.dsp.run_feature_extraction \
    --manifest data/manifests/analysis_manifest.csv \
    --run-name analysis_full \
    --feature mfcc pitch formants speech_rate_reference

# Every feature listed in FEATURE_MODULES
python -m scripts.dsp.run_feature_extraction \
    --manifest data/manifests/analysis_manifest.csv \
    --run-name analysis_full \
    --feature all
```

`--feature all` runs every feature listed in `FEATURE_MODULES` at the top of `scripts/dsp/run_feature_extraction.py`. Add a feature to that list to include it.

| Argument | Meaning |
|---|---|
| `--manifest` | Manifest CSV to process (required) |
| `--run-name` | Name for the run; reuse it to resume (required) |
| `--feature` | One or more feature names, or `all` (required) |
| `--processed-root` | Folder `processed_filepath` values are relative to (default `data/processed`) |
| `--results-root` | Folder where run folders are created (default `results/dsp/runs`) |
| `--config-dir` | Folder with `<feature>.json` and `speech_mask.json` (default `configs/dsp`) |
| `--retry-errors`, `--fail-fast` | See above |

Each feature gets its own folder, `results/dsp/runs/<run-name>/<feature>/`, containing `run_config.json`, `manifest_snapshot.csv` and `features.csv` (one row per recording: `status`, the feature columns and timing). When several features are run, a feature that crashes is reported and the others still run; a summary is printed at the end.

### 7.4 Reproducing the full experiment

The notebooks expect the following run names, so use these if you want to run them without changing any paths:

```bash
# ASR: one run per Whisper checkpoint (whisper-base is the primary model)
python -m scripts.asr.run_transcription --manifest data/manifests/analysis_manifest.csv --run-name analysis_tiny_long  --model openai/whisper-tiny
python -m scripts.asr.run_transcription --manifest data/manifests/analysis_manifest.csv --run-name analysis_base_long  --model openai/whisper-base
python -m scripts.asr.run_transcription --manifest data/manifests/analysis_manifest.csv --run-name analysis_small_long --model openai/whisper-small
python -m scripts.asr.run_transcription --manifest data/manifests/analysis_manifest.csv --run-name analysis_med_long   --model openai/whisper-medium

# Score each run
python -m scripts.asr.run_scoring --run-dir results/asr/runs/analysis_tiny_long
python -m scripts.asr.run_scoring --run-dir results/asr/runs/analysis_base_long
python -m scripts.asr.run_scoring --run-dir results/asr/runs/analysis_small_long
python -m scripts.asr.run_scoring --run-dir results/asr/runs/analysis_med_long

# DSP features
python -m scripts.dsp.run_feature_extraction --manifest data/manifests/analysis_manifest.csv --run-name analysis_full --feature mfcc pitch formants speech_rate_reference
```

For `decode_mode_comparison.ipynb`, SAA recordings of 30 s or less are transcribed in both decode modes with `whisper-base`:

```bash
python -m scripts.experiments.build_duration_subset \
    --input data/manifests/study_manifest.csv \
    --max-duration 30 \
    --dataset saa
# writes data/manifests/study_under_30s_saa.csv

python -m scripts.asr.run_transcription --manifest data/manifests/study_under_30s_saa.csv --run-name decode_base_long  --model openai/whisper-base --decode-mode sequential_long_form
python -m scripts.asr.run_transcription --manifest data/manifests/study_under_30s_saa.csv --run-name decode_base_short --model openai/whisper-base --decode-mode short_form

python -m scripts.asr.run_scoring --run-dir results/asr/runs/decode_base_long
python -m scripts.asr.run_scoring --run-dir results/asr/runs/decode_base_short
```


## 8. Notebooks

The notebooks are in `notebooks/` and read the run outputs in `results/`. Those outputs are included in the repo (see Quick start), so every notebook can be run without rerunning the pipelines. They are meant to be opened from inside the `notebooks/` folder, because their paths (e.g. `../results/asr/`) are relative to it. The paths and settings are collected in a configuration cell at the top of each notebook. Figures and tables are saved next to the notebook (`figures/`, `wer_outputs/`, `analysis_outputs/`).

| Notebook | What it does | Needs |
|---|---|---|
| `whisper_model_selection.ipynb` | Compares the `tiny`, `base`, `small` and `medium` checkpoints (speed vs accuracy, accent ranking, agreement between models, the WER = 0 floor) to justify using `whisper-base` as the primary model | Scored runs for all four checkpoints |
| `decode_mode_comparison.ipynb` | Compares short-form and long-form decoding on SAA recordings under 30 s to justify using long-form decoding for every recording | `decode_base_short`, `decode_base_long`, and `analysis_base_long` |
| `normalization_tokeniser.ipynb` | Checks that the choice of Whisper tokenizer (eight checkpoints) used for text normalisation does not change the normalised text or WER, so a single fixed tokenizer can be used | `analysis_base_long` (transcriptions) |
| `punctuation_removal.ipynb` | Compares scoring with and without removal of AfriSpeech's spoken punctuation words, and shows what the removal does to WER | `scored_punc_kept.csv` and `scored_punc_removed.csv` in `analysis_base_long` |
| `WER_by_Accent.ipynb` | WER per accent group: pooled and mean WER, Hindi and African vs American, error types, differences within the African groups, and the same accents in SAA and AfriSpeech | `analysis_base_long` scored results; optionally `analysis_outputs/analysis_master_table.csv` |
| `Accent_WER_Analysis.ipynb` | The main statistical analysis. Stage 1 finds which acoustic descriptors differ from the American baseline (Mann–Whitney, Cliff's δ, Benjamini–Hochberg). Stage 2 relates each speaker's distance from the baseline to their WER (Spearman). It also looks at how much of the WER speech rate explains and compares speakers at the same speaking rate | `analysis_base_long` scored results and the `mfcc`, `pitch`, `formants` and `speech_rate_reference` features from `analysis_full` |

`Accent_WER_Analysis.ipynb` writes `analysis_outputs/analysis_master_table.csv` (one row per analysed recording with WER and all features). Run it before `WER_by_Accent.ipynb` if you want the WER notebook to use exactly the same recordings as the acoustic analysis; setting `ANALYSIS_SAMPLE = None` in that notebook uses every scored recording instead.


## 9. Testing

All tests should be run from the repository root with the virtual environment activated.

```bash
python -m pytest -m "not integration" -v
```

Most tests use synthetic data. Real-data sanity checks are skipped if the required local dataset or manifest is not available. The real Whisper integration tests are skipped by default (see below).

The current test suite covers:

| Test file | Purpose |
| --- | --- |
| `test_afrispeech_loader.py` | AfriSpeech dataset loading |
| `test_analysis_manifest.py` | Analysis-manifest generation and validation |
| `test_audio_io.py` | Processed-audio loading and validation |
| `test_dev_manifest.py` | Development-manifest generation |
| `test_feature_extraction_pipeline.py` | Resumable DSP extraction pipeline: checkpointing, validation and error handling (with a fake feature module) |
| `test_formants.py` | Formant (F1–F3) extraction |
| `test_manifest_checks.py` | Shared manifest and results-table checks |
| `test_metrics.py` | WER calculation and substitution/deletion/insertion counts |
| `test_mfcc.py` | Speech-masked MFCC features |
| `test_normalization.py` | Transcript normalization |
| `test_pitch.py` | Pitch (F0) extraction |
| `test_preprocessing_audio.py` | Audio preprocessing functionality |
| `test_preprocessing_exclusion.py` | Preprocessing exclusion rules and excluded-sample handling |
| `test_preprocessing_quality.py` | Audio quality checks used during preprocessing |
| `test_run_feature_extraction.py` | Feature-extraction CLI: feature selection, config loading, resume checks and multi-feature runs |
| `test_run_management.py` | Shared run-folder and checkpoint helpers |
| `test_run_scoring.py` | Scoring CLI/run orchestration, configuration and output-file handling |
| `test_run_transcription.py` | Transcription CLI/run orchestration, experiment configuration and resume validation |
| `test_saa_loader.py` | Speech Accent Archive loading |
| `test_scoring_pipeline.py` | Scoring-pipeline validation, normalization/WER orchestration, statuses and error handling |
| `test_select_afrispeech.py` | Selection and filtering of AfriSpeech samples |
| `test_select_analysis.py` | Selection of samples for the analysis dataset |
| `test_select_saa.py` | Selection and filtering of Speech Accent Archive samples |
| `test_speech_mask.py` | Shared DSP speech-mask calculation, temporal smoothing and interval handling |
| `test_speech_rate_acoustic.py` | Transcript-free acoustic speech rate |
| `test_speech_rate_reference.py` | Reference-transcript speech rate and pause features |
| `test_study_manifest.py` | Study-manifest generation |
| `test_text_syllables.py` | Transcript syllable counting used by the reference speech rate |
| `test_transcription_pipeline.py` | Manifest-level transcription, checkpointing, resume/retry behaviour and error handling |
| `test_transcription.py` | Whisper transcription, short-form/long-form decoding, timing, configuration and edge cases |
| `test_validation.py` | Dataset/manifest validation |

### Run a specific test file

For example:

```bash
python -m pytest tests/test_metrics.py
```

### Whisper integration tests

Most tests in `test_transcription.py` use mocked Hugging Face processor/model objects. This keeps the unit tests fast and means they do not require a GPU or model download.

`test_transcription.py` also contains integration tests that use the real Hugging Face `openai/whisper-tiny` model. These tests verify that the installed `transformers` version accepts the preprocessing and generation arguments used by the project.

The integration tests are skipped by default because they are slower and may need to download/cache the Whisper checkpoint on the first run.

#### Git Bash / Linux / macOS

Enable and run only the integration tests:

```bash
export RUN_WHISPER_INTEGRATION=1
python -m pytest -m integration -v
```

Remove the environment variable afterwards:

```bash
unset RUN_WHISPER_INTEGRATION
```

#### Windows PowerShell

Enable and run only the integration tests:

```powershell
$env:RUN_WHISPER_INTEGRATION="1"
python -m pytest -m integration -v
```

Remove the environment variable afterwards:

```powershell
Remove-Item Env:RUN_WHISPER_INTEGRATION
```

### Run the complete test suite, including integration tests

#### Git Bash / Linux / macOS

```bash
export RUN_WHISPER_INTEGRATION=1
python -m pytest -v
```

#### Windows PowerShell

```powershell
$env:RUN_WHISPER_INTEGRATION="1"
python -m pytest -v
```


## 10. Project repo structure
```
project/
│
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml                  # package settings and pytest configuration
│
├── configs/
│   ├── afrispeech_acquisition.json # AfriSpeech accents, splits and dataset revision
│   └── dsp/                        # one JSON config per DSP feature + speech_mask.json
│
├── src/                            # contains the core code and functions
│   │ 
│   ├── common/                     # audio loading, frame energy, manifest checks, run folders/checkpointing
│   │
│   ├── data/                       # SAA/AfriSpeech loaders, selection rules, schema, validation
│   │
│   ├── preprocessing/              # audio conversion, QC measurements, exclusion rules
│   │
│   ├── asr/                        # Whisper transcription, normalization, WER metrics, pipelines
│   │
│   └── dsp/
│       ├── speech_mask.py          # shared speech-activity mask
│       ├── text_syllables.py       # syllable counting for transcripts
│       ├── extraction_pipeline.py  # resumable feature-extraction pipeline
│       └── features/               # mfcc, pitch, formants, speech_rate_*, rms_test
│
├── scripts/                        # run with: python -m scripts.<folder>.<script>
│   ├── data/                       # data acquisition and manifest-building scripts
│   ├── preprocessing/              # audio preprocessing and QC scripts
│   ├── asr/                        # run_transcription.py, run_scoring.py
│   ├── dsp/                        # run_feature_extraction.py + inspection/tuning scripts
│   ├── experiments/                # build_duration_subset.py (short- vs long-form comparison)
│   └── inspection/                 # exploratory data/audio inspection scripts
│
├── notebooks/                      # analysis and justification notebooks (see section 8)
│   └── analysis_outputs/           # tables/figures written by Accent_WER_Analysis.ipynb
│
├── data/
│   ├── raw/                        # GITIGNORED
│   ├── processed/                  # GITIGNORED
│   └── manifests/                  # GITIGNORED
│       └── .csv
│
├── results/                        # GITIGNORED
│   ├── asr/runs/<run-name>/        # run_config.json, manifest_snapshot.csv, transcriptions.csv, scored.csv
│   └── dsp/runs/<run-name>/<feature>/   # run_config.json, manifest_snapshot.csv, features.csv
│
├── tests/
│   └── .py files
│
└── third_party/                    # licenses / references for third party adapted code

```

### Other scripts

These are not part of the main pipeline. They were used to make and justify design choices (dataset selection, QC thresholds, DSP settings) and are kept for reference. Each has a docstring at the top explaining what it does.

| Folder | Scripts |
|---|---|
| `scripts/inspection/` | `inspect_saa.py`, `inspect_saa_candidates.py`, `inspect_afrispeech.py`, `audit_afrispeech_speakers.py` (dataset exploration); `inspect_audio.py`, `inspect_audio_followup.py`, `check_wada_saturation.py` (audio properties and QC) |
| `scripts/dsp/` | `inspect_speech_mask.py`, `check_speech_mask_prevalence.py` (speech-mask tuning); `inspect_pitch.py`, `inspect_formants.py` (pitch and formant settings) |
| `scripts/experiments/` | `build_duration_subset.py` (duration-filtered manifest) |


## 11. Using the shared audio loader and speech mask

### Load processed audio

Processed audio should be loaded with `load_processed_audio()`:

```python
from pathlib import Path

import pandas as pd

from src.common.audio_io import (
    SAMPLE_RATE,
    load_processed_audio,
)


PROCESSED_ROOT = Path("data/processed")

manifest = pd.read_csv(
    "data/manifests/analysis_manifest.csv",
    keep_default_na=False,
)

row = manifest.iloc[0]

audio = load_processed_audio(
    PROCESSED_ROOT,
    row["processed_filepath"],
)
```

`load_processed_audio()` returns the validated processed waveform as a one-dimensional NumPy array:

```text
np.ndarray
```

`audio` therefore contains the audio samples for the recording. The loader checks the expected processed-audio format and does not resample, normalise or trim the signal.

### Get the shared speech intervals

Use `speech_intervals()` to obtain the speech-active regions for the recording:

```python
import json
from pathlib import Path

from src.dsp.speech_mask import speech_intervals


CONFIG_PATH = Path("configs/dsp/speech_mask.json")

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    speech_mask_config = json.load(file)

intervals = speech_intervals(
    audio=audio,
    sample_rate=SAMPLE_RATE,
    config=speech_mask_config,
)
```

`speech_intervals()` returns a Python list of `(start_s, end_s)` tuples:

```text
list[tuple[float, float]]
```

For example:

```text
[
    (0.42, 3.18),
    (3.31, 7.84),
]
```

Each tuple represents one detected speech-active region, with the start and end times given in seconds.

### Apply the speech mask to feature timestamps

Feature extractors should run on the full waveform first. Their frame timestamps can then be matched to the shared speech intervals:

```python
from src.dsp.speech_mask import speech_mask_for_times

speech_mask = speech_mask_for_times(
    intervals=intervals,
    times=feature_times,
)

speech_features = features[speech_mask]
```

`speech_mask_for_times()` returns a one-dimensional Boolean NumPy array:

```text
np.ndarray[bool]
```

It has one Boolean value for each value in `feature_times`:

- `True` — the feature frame falls inside a speech-active interval.
- `False` — the feature frame falls outside the speech-active intervals.

The Boolean mask can then be used to select the corresponding speech frames from the feature array:

```python
speech_features = features[speech_mask]
```

This allows MFCC, pitch, formant and other DSP features to use the same speech definition even when their frame grids differ.

### Adding a new feature

A feature is a module in `src/dsp/features/<name>.py` with a matching `configs/dsp/<name>.json`. The module must define `FEATURE_NAME` (the same as the file name), `FEATURE_VERSION`, `feature_columns(config)` (the output column names) and `extract(audio, sample_rate, row, config)` (returns a dict with one value per column). `rms_test.py` is the smallest working example. Bump `FEATURE_VERSION` whenever the calculation changes, so that old runs are not resumed with the new code. To include the feature in `--feature all`, add it to `FEATURE_MODULES` in `scripts/dsp/run_feature_extraction.py`.


## 12. Reproducibility and limitations

### Reproducibility

- **Pinned software and data:** package versions are pinned in `requirements.txt`, and the AfriSpeech dataset revision is fixed in `configs/afrispeech_acquisition.json`.
- **Fixed seeds:** the random selection of one AfriSpeech recording per source ID uses a fixed seed (42), and so does the bootstrap in `Accent_WER_Analysis.ipynb`.
- **Run records:** every ASR and DSP run stores its settings, hardware, software versions and a hash of the manifest it started from in `run_config.json`, and resuming is refused if any of these have changed.
- **Hardware and precision:** Whisper transcripts can differ slightly between GPUs, between FP16 and FP32, and between `transformers` versions. To compare against the stored results, use the settings recorded in each run's `run_config.json`.

### Limitations

- **AfriSpeech is a partial replication only.** Each speaker reads different, often medical, text; recordings come from many devices; and distances are measured against SAA American speakers recorded under SAA conditions. The same accents get quite different WER and distances in the two corpora.
- **Small African subgroups in SAA.** After cleaning, the SAA African groups are small (for example 5 Yoruba, 7 Kiswahili and 7 Tigrigna speakers), so the African speakers are analysed as one group and group-level estimates are uncertain.
- **AfriSpeech speakers.** AfriSpeech source IDs have not been independently verified as unique speakers.
- **One ASR model.** The main results use `whisper-base`; the other checkpoints are only used to justify that choice.
- **Excluded recordings.** Four AfriSpeech recordings with broken transcriptions, and recordings with implausible F1 tracks, are removed in the analysis notebook (see `EXCLUDE_IDS` and the F1 thresholds in its settings cell).


## 13. Data sources, licences and citations

- **Speech Accent Archive:** Weinberger, S. (2013). *Speech Accent Archive.* George Mason University. Downloaded from the Kaggle copy: https://www.kaggle.com/datasets/rtatman/speech-accent-archive
- **AfriSpeech-200:** Olatunji, T. et al. (2023). *AfriSpeech-200: Pan-African Accented Speech Dataset for Clinical and General Domain ASR.* Hugging Face: `intronhealth/afrispeech-200`.
- **Whisper:** Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* Checkpoints from Hugging Face (`openai/whisper-*`).
- **Third-party code:** the audio quality measurements in `src/preprocessing/quality.py` are adapted from the [URGENT 2024 analysis code](https://github.com/urgent-challenge/urgent2024_analysis) (MIT licence). Its licence is kept in `third_party/urgent2024_analysis_LICENSE`, and the changes made are listed in the comments in `quality.py`.

The datasets are not included in the repo and are subject to their own licences and terms of use.


## 14. Troubleshooting

- **`ModuleNotFoundError: No module named 'src'` (or `scripts`):** run commands from the project root with `python -m ...` (e.g. `python -m scripts.asr.run_scoring`), not `python scripts/asr/run_scoring.py`.
- **"Cannot resume this run because its configuration has changed":** the manifest, model, decode mode, device, precision or software versions differ from the first time the run was started. Use a new `--run-name` for the new experiment.
- **Whisper is very slow:** check that PyTorch can see the GPU (`python -c "import torch; print(torch.cuda.is_available())"`). On a CPU, use the smaller checkpoints or the dev manifest.
- **Out of GPU memory:** try `--device cpu`, a smaller checkpoint, or `--fp32` if FP16 is the problem.
- **Failed samples in a run:** they are recorded with `status = error` in the results CSV. Fix the cause and run the same command again with `--retry-errors`.
- **A notebook can't find a results file:** the notebooks must be opened from inside `notebooks/`, and run names in the configuration cell have to match the folders in `results/`.
- **Hugging Face download or login errors:** run `hf auth login` (see section 2) and check your internet connection. Models and tokenizers are downloaded the first time they are used.
