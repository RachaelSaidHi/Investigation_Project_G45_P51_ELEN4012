import numpy as np

FEATURE_NAME = "rms_test"
FEATURE_VERSION = "0"

def feature_columns(config):
    return ["rms_value"]

def extract(audio, sample_rate, row, config):
    return {"rms_value": float(np.sqrt(np.mean(audio ** 2)))}