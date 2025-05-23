#!/usr/bin/env python3
"""
Classificador Fuzzy para Dataset GTZAN
"""

import os
import pandas as pd
from pathlib import Path
from src.feature_extraction import AudioFeatureExtractor
from src.preprocessing import DataPreprocessor
from src.fuzzy_classifier import FuzzyMusicClassifier
from src.evaluation import ModelEvaluator

def main():
    # Configurações
    DATA_PATH = "data/raw/GTZAN"  # Caminho para o dataset GTZAN
    FEATURES_PATH = "data/processed/features.csv"
    MODEL_PATH = "data/models/fuzzy_classifier.joblib"
    
    print("=== CLASSIFICADOR FUZZY GTZAN ===\n")
    
    
    print("1. Extraindo features do áudio...")
    if not os.path.exists(FEATURES_PATH):
        os.makedirs("data/processed", exist_ok=True)
        extractor = AudioFeatureExtractor()
        df_features = extractor.extract_dataset_features(DATA_PATH, FEATURES_PATH)
    else:
        df_features = pd.read_csv(FEATURES_PATH)
        print(f"Features carregadas de: {FEATURES_PATH}")
    
    print(f"Dataset shape: {df_features.shape}")
    print(f"Gêneros encontrados: {df_features['genre'].unique()}")
    
    print("\n2. Preprocessando dados...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df_features)
    
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    print(f"Features: {X_train.shape[1]}")
    
    print("\n3. Treinando classificador fuzzy...")
    fuzzy_classifier = FuzzyMusicClassifier(n_features=8, n_rules=10)
    fuzzy_classifier.fit(X_train, y_train)
    
    print("\n4. Fazendo predições...")
    y_pred = fuzzy_classifier.predict(X_test)
    
    print("\n5. Avaliando modelo...")
    genre_mapping = preprocessor.get_genre_mapping()
    evaluator = ModelEvaluator(genre_mapping)
    
    metrics = evaluator.evaluate_model(y_test, y_pred)
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_performance_by_genre(y_test, y_pred)
    
    print("\n6. Salvando modelo...")
    os.makedirs("data/models", exist_ok=True)
    fuzzy_classifier.save_model(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")
    
    print(f"\n=== CONCLUÍDO ===")
    print(f"Acurácia final: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()