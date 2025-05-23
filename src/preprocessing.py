import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocessa os dados para o classificador fuzzy"""
        # Remove colunas não numéricas exceto 'genre'
        feature_cols = [col for col in df.columns 
                       if col not in ['genre', 'filename'] and 
                       df[col].dtype in ['float64', 'int64']]
        
        self.feature_columns = feature_cols
        
        # Separa features e labels
        X = df[feature_cols].copy()
        y = df['genre'].copy()
        
        # Remove valores NaN e infinitos
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Normaliza features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # Codifica labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_genre_mapping(self):
        """Retorna o mapeamento de classes"""
        return dict(enumerate(self.label_encoder.classes_))