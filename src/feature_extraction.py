import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path):
        """Extrai features de um arquivo de áudio"""
        try:
           
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            features = {}
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features.update({
                f'mfcc_mean_{i}': np.mean(mfccs[i]) for i in range(self.n_mfcc)
            })
            features.update({
                f'mfcc_std_{i}': np.std(mfccs[i]) for i in range(self.n_mfcc)
            })
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.update({
                f'chroma_mean_{i}': np.mean(chroma[i]) for i in range(12)
            })
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            return features
            
        except Exception as e:
            print(f"Erro ao processar {audio_path}: {e}")
            return None
    
    def extract_dataset_features(self, data_path, output_path=None):
        """Extrai features de todo o dataset GTZAN"""
        data_path = Path(data_path)
        features_list = []
        
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        for genre in tqdm(genres, desc="Processando gêneros"):
            genre_path = data_path / genre
            if not genre_path.exists():
                continue
                
            for audio_file in tqdm(list(genre_path.glob("*.wav")), 
                                 desc=f"Processando {genre}", leave=False):
                features = self.extract_features(audio_file)
                if features:
                    features['genre'] = genre
                    features['filename'] = audio_file.name
                    features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Features salvos em: {output_path}")
        
        return df