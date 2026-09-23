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
**The dataset requires approximately 920 MB of disk space after extraction**

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
python -m scripts.acquire_afrispeech
```

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


## 5. Build the manifests

Run these three scripts in order from the project root:

```bash
# Step 1: Build the full manifest from all locally available datasets
python -m scripts.build_full_manifest

# Step 2: Apply the study selection rules
python -m scripts.build_study_manifest

# Step 3: Build the small dev subset for quick iteration
python -m scripts.build_dev_manifest
```

This produces three files in `data/manifests/`:

| File | Contents | Use |
|---|---|---|
| `full_manifest.csv` | All locally available SAA and AfriSpeech samples in the common manifest format | Input to the study-selection step |
| `study_manifest.csv` | Samples selected for the investigation using the dataset-specific study rules | The actual experimental dataset |
| `dev_manifest.csv` | Small subset containing up to 1 female and 1 male speaker per dataset/accent group, with one recording per selected speaker | Quick pipeline development and testing |

The full manifest contains all the data available locally, while the study manifest contains only the samples selected for the experiment.

## 6. Verify

```bash
python -m pytest -v
```

All tests should pass.

Most tests use synthetic data. Real-data sanity checks are skipped if the required local dataset or manifest is not available.

## 7. What the manifests contain

The manifests use the same common schema. Important columns for the ASR and analysis pipeline include:

- `sample_id` — unique identifier for each audio sample
- `dataset` — source dataset (`saa` or `afrispeech`)
- `source_split` — original dataset split where applicable
- `filepath` — path to the audio file relative to `data/raw/`
- `speaker_id` — unique speaker identifier within the source dataset
- `accent_raw` — accent-related label provided by the source dataset
- `accent_group` — study accent category used for analysis
- `reference_transcript` — correct transcript for WER computation
- `sex` — speaker sex metadata
- `age` — exact age where available
- `age_group` — age range where available
- `domain` — speech domain or recording context

SAA contains one recording per speaker. AfriSpeech may contain multiple recordings from the same speaker.

To load audio for a given manifest row:

```python
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/raw")

manifest = pd.read_csv(
    "data/manifests/study_manifest.csv",
    keep_default_na=False,
)

row = manifest.iloc[0]
audio_path = DATA_ROOT / row["filepath"]
```

## Project Repo Structure
```
project/
│
├── README.md
├── .gitignore
├── requirements.txt
├── src/                        # contains the core code and functions
│   | 
│   ├── asr/
│   │   └── .py files
│   │
│   ├── dsp/
│   │   └── .py files
│   │
│   └── analysis/
│       └── .py files
│
├── scripts/                    # like python notebooks maybe
│   └── python notebooks
│
├── data/
│   ├── README.md
│   ├── raw/                    # GITIGNORED
│   ├── preprocessed/           # GITIGNORED
│   └── manifests/
│       └── .csv
│
├── results/
│   ├── asr/                 
│   ├── dsp/
│   └── analysis/
│
└── tests/
    └── .py files

```
