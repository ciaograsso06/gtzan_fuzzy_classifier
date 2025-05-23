import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score)

class ModelEvaluator:
    def __init__(self, genre_mapping):
        self.genre_mapping = genre_mapping
    
    def evaluate_model(self, y_true, y_pred):
        """Avalia o desempenho do modelo"""
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("=== RESULTADOS DO CLASSIFICADOR FUZZY ===")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Relatório detalhado
        genre_names = [self.genre_mapping[i] for i in sorted(self.genre_mapping.keys())]
        print("\n=== RELATÓRIO DETALHADO ===")
        print(classification_report(y_true, y_pred, target_names=genre_names))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(10, 8)):
        """Plota matriz de confusão"""
        cm = confusion_matrix(y_true, y_pred)
        genre_names = [self.genre_mapping[i] for i in sorted(self.genre_mapping.keys())]
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=genre_names, yticklabels=genre_names)
        plt.title('Matriz de Confusão - Classificador Fuzzy')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_performance_by_genre(self, y_true, y_pred):
        """Plota performance por gênero"""
        genre_names = [self.genre_mapping[i] for i in sorted(self.genre_mapping.keys())]
        report = classification_report(y_true, y_pred, target_names=genre_names, output_dict=True)
        
        # Extrai métricas por classe
        genres = list(report.keys())[:-3]  # Remove 'accuracy', 'macro avg', 'weighted avg'
        precision_scores = [report[genre]['precision'] for genre in genres]
        recall_scores = [report[genre]['recall'] for genre in genres]
        f1_scores = [report[genre]['f1-score'] for genre in genres]
        
        # Plot
        x = np.arange(len(genres))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision_scores, width, label='Precisão', alpha=0.8)
        ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Gêneros Musicais')
        ax.set_ylabel('Score')
        ax.set_title('Performance por Gênero Musical')
        ax.set_xticks(x)
        ax.set_xticklabels(genres, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()