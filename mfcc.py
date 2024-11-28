import librosa
import numpy as np
import pandas as pd

def process_mfcc(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Convert to DataFrame
    mfcc_df = pd.DataFrame(mfccs.T)
    return mfcc_df
