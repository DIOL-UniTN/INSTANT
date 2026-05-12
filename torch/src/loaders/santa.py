import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import librosa
import torch
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = "./data/santa/"
SETS = ['train', 'test']
class SANTALoader: 
    def __init__(self, batch_size: int, debug_level: bool):
        self.name = 'santa'
        self.debug = debug_level > 1
        self.data_dir = Path(DATA_DIR)

        audio_directory = self.data_dir/"recordings"
        metadata_file = self.data_dir/"metadata1.csv"

        if (not (self.data_dir/"santa_train.pt").exists() or 
            not (self.data_dir/"santa_test.pt").exists()):
            print("Creating dataset features...")
            x_train, x_test, y_train, y_test = create_dataset(audio_directory, metadata_file)
            torch.save((x_train, y_train), self.data_dir/"santa_train.pt")
            torch.save((x_test, y_test), self.data_dir/"santa_train.pt")
        else:
            x_train, y_train = torch.load(self.data_dir/"santa_train.pt")
            x_test, y_test = torch.load(self.data_dir/"santa_test.pt")

        trainset = TensorDataset(x_train, y_train)
        testset = TensorDataset(x_test, y_test)

        num_workers = 0 if self.debug else 8
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size
        self.in_chan = 1
        self.in_size = (1, 37)
        self.out_dim = 18

def create_dataset(audio_dir: Path, metadata_file):
    # Load the metadata
    metadata_df = pd.read_csv(metadata_file)
    features = []  # Features
    labels = []  # Labels (age)

    print(f"Total entries in metadata: {len(metadata_df)}")

    # Iterate through each row in the metadata
    for index, row in metadata_df.iterrows():
        print(f"\nProcessing index {index}")
        recording_id = row['RECORDING']  # Get the recording ID

        # Construct the path to the audio file based on the RECORDING ID
        filename = f"0{recording_id}.mp3" if recording_id < 9 else f"{recording_id}.mp3"
        audio_file = audio_dir / filename
        print(f"Attempting to process file: {audio_file}")

        # Check if file exists
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            continue

        # Extract features from the audio file
        feature = extract_features(audio_file)
        label = index

        # Append features and age to the lists
        features.append(feature)
        labels.append(label)

        print(f"Successfully processed file: {audio_file}")
        print(f"Features shape: {feature.shape}")
        print(f"Features sample: {feature[:10]}")  # Print first 10 features
        print(f"Person id: {label}")

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    print(f"\nFinal dataset shape: features: {features.shape}, labels: {labels.shape}")
    print(f"Total successfully processed files: {len(features)}")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert al to torch
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    return x_train, x_test, y_train, y_test

def extract_features(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Extract MFCCs (Mel Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)

    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = np.mean(chroma.T, axis=0)

    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast = np.mean(spectral_contrast.T, axis=0)

    # Extract Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid = np.mean(spectral_centroid)

    # Extract Signal Energy
    signal_energy = np.sum(y ** 2)

    # Extract Formants (using librosa's piptrack for pitch)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    formants = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Ensure valid pitch values
            formants.append(pitch)
    formants = np.mean(formants[:5]) if len(formants) > 0 else 0  # Taking the average of the first 5 formants

    # Extract Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate = np.mean(zero_crossing_rate)

    # Extract Peak Value
    peak_value = np.max(np.abs(y))

    # Ensure all features are 1D arrays and concatenate them
    features = np.concatenate([mfccs, chroma, spectral_contrast, 
                               [spectral_centroid], [signal_energy], 
                               [formants], [zero_crossing_rate], 
                               [peak_value]
                               ])

    return features

