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
    # Configurações - Ajustado para a estrutura correta do GTZAN
    DATA_PATH = "data/genres_original"  # Caminho correto para o dataset GTZAN
    FEATURES_PATH = "data/processed/features.csv"
    MODEL_PATH = "data/models/fuzzy_classifier.joblib"
    
    print("=== CLASSIFICADOR FUZZY GTZAN ===\n")
    
    # Verifica se o dataset existe
    if not os.path.exists(DATA_PATH):
        print(f"ERRO: Dataset não encontrado em {DATA_PATH}")
        print("Por favor, certifique-se de que o dataset GTZAN está na pasta correta.")
        return
    
    # 1. Extração de Features
    print("1. Extraindo features do áudio...")
    
    # Verifica se o arquivo de features já existe e não está vazio
    need_extraction = True
    if os.path.exists(FEATURES_PATH):
        try:
            # Tenta ler o arquivo para verificar se não está vazio/corrompido
            df_test = pd.read_csv(FEATURES_PATH)
            if len(df_test) > 0 and 'genre' in df_test.columns:
                df_features = df_test
                print(f"Features carregadas de: {FEATURES_PATH}")
                need_extraction = False
            else:
                print("Arquivo de features existe mas está vazio ou corrompido. Reextraindo...")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
            print(f"Erro ao ler features existentes: {e}")
            print("Extraindo features novamente...")
    
    if need_extraction:
        print("Iniciando extração de features... (isso pode demorar alguns minutos)")
        os.makedirs("data/processed", exist_ok=True)
        
        extractor = AudioFeatureExtractor()
        df_features = extractor.extract_dataset_features(DATA_PATH, FEATURES_PATH)
        
        if df_features is None or len(df_features) == 0:
            print("ERRO: Falha na extração de features. Verifique o dataset.")
            return
    
    print(f"Dataset shape: {df_features.shape}")
    print(f"Gêneros encontrados: {sorted(df_features['genre'].unique())}")
    print(f"Arquivos por gênero:")
    for genre in sorted(df_features['genre'].unique()):
        count = len(df_features[df_features['genre'] == genre])
        print(f"  {genre}: {count} arquivos")
    
    # Verifica se temos dados suficientes
    if len(df_features) < 50:
        print("AVISO: Poucos arquivos encontrados. Verifique se o dataset está completo.")
    
    # 2. Preprocessamento
    print("\n2. Preprocessando dados...")
    preprocessor = DataPreprocessor()
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df_features)
    except Exception as e:
        print(f"ERRO no preprocessamento: {e}")
        return
    
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    print(f"Features: {X_train.shape[1]}")
    
    # 3. Treinamento do Classificador Fuzzy
    print("\n3. Treinando classificador fuzzy...")
    try:
        fuzzy_classifier = FuzzyMusicClassifier(n_features=8, n_rules=10)
        fuzzy_classifier.fit(X_train, y_train)
        print("Treinamento concluído com sucesso!")
    except Exception as e:
        print(f"ERRO no treinamento: {e}")
        return
    
    # 4. Predição
    print("\n4. Fazendo predições...")
    try:
        y_pred = fuzzy_classifier.predict(X_test)
        print(f"Predições realizadas para {len(y_pred)} amostras")
    except Exception as e:
        print(f"ERRO na predição: {e}")
        return
    
    # 5. Avaliação
    print("\n5. Avaliando modelo...")
    try:
        genre_mapping = preprocessor.get_genre_mapping()
        evaluator = ModelEvaluator(genre_mapping)
        
        metrics = evaluator.evaluate_model(y_test, y_pred)
        
        # Plots (com tratamento de erro para ambientes sem display)
        try:
            evaluator.plot_confusion_matrix(y_test, y_pred)
            evaluator.plot_performance_by_genre(y_test, y_pred)
        except Exception as plot_error:
            print(f"Aviso: Não foi possível gerar gráficos: {plot_error}")
            
    except Exception as e:
        print(f"ERRO na avaliação: {e}")
        return
    
    # 6. Salvando modelo
    print("\n6. Salvando modelo...")
    try:
        os.makedirs("data/models", exist_ok=True)
        fuzzy_classifier.save_model(MODEL_PATH)
        print(f"Modelo salvo em: {MODEL_PATH}")
    except Exception as e:
        print(f"ERRO ao salvar modelo: {e}")
    
    print(f"\n=== CONCLUÍDO ===")
    print(f"Acurácia final: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()