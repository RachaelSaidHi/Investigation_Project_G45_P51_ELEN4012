# Investigation_Project_G45_P51_ELEN4012

## Setting Up Whisper
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
4. Install remaining project dependencies
```
python -m pip install -r requirements.txt
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
