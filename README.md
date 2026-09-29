# Investigation_Project_G45_P51_ELEN4012


## 1. Clone and install

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## 2. Set Up Whisper
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

The accent and split combinations that will be downloaded are specified in:

```text
configs/afrispeech_acquisition.json
```

The script downloads the required transcript metadata and audio and stores them under:

```text
data/raw/afrispeech/
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

### Quality control

Recordings are excluded if their duration is outside 5–60 seconds or they contain less than 0.05% spectral energy above 4 kHz. WADA-SNR is measured for diagnostic purposes but is not used for exclusion.

### Analysis selection

The analysis manifest retains all QC-passing SAA recordings and one randomly selected QC-passing recording per AfriSpeech source ID, using a fixed seed.

The development manifest contains a smaller subset of the analysis recordings for pipeline testing.


## 6. Verify

```bash
python -m pytest -v
```

All tests should pass.

Most tests use synthetic data. Real-data sanity checks are skipped if the required local dataset or manifest is not available.

## 7. What the manifests contain

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

To locate the processed audio for a given manifest row:

```python
from pathlib import Path
import pandas as pd

PROCESSED_ROOT = Path("data/processed")

manifest = pd.read_csv(
    "data/manifests/analysis_manifest.csv",
    keep_default_na=False,
)

row = manifest.iloc[0]
audio_path = PROCESSED_ROOT / row["processed_filepath"]
```

## Project Repo Structure
```
project/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── configs/                     # acquisition and other project configurations
│
├── src/                        # contains the core code and functions
│   | 
│   ├── data/
│   │   └── .py files
│   │
│   ├── preprocessing/
│   │   └── .py files
│   │
│   ├── asr/
│   │   └── .py files
│   │
│   ├── dsp/
│   │   └── .py files
│   │
│   └── analysis/
│       └── .py files
│
├── scripts/
│   ├── inspection/              # exploratory data/audio inspection scripts
│   ├── data/                    # data acquisition and manifest-building scripts
│   ├── preprocessing/           # audio preprocessing and QC scripts
│   └── asr/
│
├── data/
│   ├── raw/                    # GITIGNORED
│   ├── processed/              # GITIGNORED
│   └── manifests/              # GITIGNORED
│       └── .csv
│
├── results/
│   ├── asr/                 
│   ├── dsp/
│   └── analysis/
│
├── tests/
│   └── .py files
│
└── third_party/                # licenses / references for third party adapted code

```
