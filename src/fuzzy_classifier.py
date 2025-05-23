import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import joblib

class FuzzyMusicClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features=5, n_rules=10):
        self.n_features = n_features
        self.n_rules = n_rules
        self.fuzzy_system = None
        self.feature_names = None
        self.classes_ = None
        self.centers_ = None
        
    def _select_best_features(self, X, y):
        """Seleciona as melhores features usando análise de variância"""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        selector = SelectKBest(score_func=f_classif, k=self.n_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_indices = selector.get_support(indices=True)
        self.feature_names = [f"feature_{i}" for i in selected_indices]
        
        return X_selected, selected_indices
    
    def _create_membership_functions(self, X):
        """Cria funções de pertinência para cada feature"""
        self.antecedents = []
        
        for i in range(self.n_features):
            feature_range = [X[:, i].min() - 0.1, X[:, i].max() + 0.1]
            
            antecedent = ctrl.Antecedent(
                np.arange(feature_range[0], feature_range[1], 
                         (feature_range[1] - feature_range[0])/100),
                f'feature_{i}'
            )
            
            antecedent['low'] = fuzz.trimf(antecedent.universe, 
                                         [feature_range[0], feature_range[0], 
                                          np.percentile(X[:, i], 33)])
            antecedent['medium'] = fuzz.trimf(antecedent.universe,
                                            [np.percentile(X[:, i], 25),
                                             np.percentile(X[:, i], 50),
                                             np.percentile(X[:, i], 75)])
            antecedent['high'] = fuzz.trimf(antecedent.universe,
                                          [np.percentile(X[:, i], 67),
                                           feature_range[1], feature_range[1]])
            
            self.antecedents.append(antecedent)
    
    def _create_output_variable(self, n_classes):
        """Cria variável de saída para as classes"""
        self.consequent = ctrl.Consequent(np.arange(0, n_classes, 1), 'genre')
        
        for i in range(n_classes):
            self.consequent[f'class_{i}'] = fuzz.trimf(
                self.consequent.universe, [i-0.4, i, i+0.4]
            )
    
    def _generate_rules(self, X, y):
        """Gera regras fuzzy baseadas nos dados de treino"""
        n_classes = len(np.unique(y))
        rules = []
        
        self.centers_ = []
        for class_idx in range(n_classes):
            class_mask = y == class_idx
            if np.sum(class_mask) > 0:
                center = np.mean(X[class_mask], axis=0)
                self.centers_.append(center)
            else:
                self.centers_.append(np.zeros(self.n_features))
        
        for class_idx in range(n_classes):
            center = self.centers_[class_idx]
            
            
            rule_antecedents = []
            for feat_idx in range(self.n_features):
                feat_val = center[feat_idx]
                
                
                if feat_val <= np.percentile(X[:, feat_idx], 33):
                    term = 'low'
                elif feat_val <= np.percentile(X[:, feat_idx], 67):
                    term = 'medium'
                else:
                    term = 'high'
                
                rule_antecedents.append(self.antecedents[feat_idx][term])
            
            if len(rule_antecedents) > 1:
                rule_condition = rule_antecedents[0]
                for ant in rule_antecedents[1:]:
                    rule_condition = rule_condition & ant
            else:
                rule_condition = rule_antecedents[0]
            
            rule = ctrl.Rule(rule_condition, self.consequent[f'class_{class_idx}'])
            rules.append(rule)
        
        return rules
    
    def fit(self, X, y):
        """Treina o classificador fuzzy"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        X_selected, self.selected_indices_ = self._select_best_features(X, y)
        
        self._create_membership_functions(X_selected)
        
        self._create_output_variable(n_classes)
        
        
        rules = self._generate_rules(X_selected, y)
        
        self.fuzzy_system = ctrl.ControlSystem(rules)
        self.fuzzy_sim = ctrl.ControlSystemSimulation(self.fuzzy_system)
        
        return self
    
    def predict(self, X):
        """Faz predições usando o sistema fuzzy"""
        if self.fuzzy_sim is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        predictions = []
        
        X_selected = X.iloc[:, self.selected_indices_] if hasattr(X, 'iloc') else X[:, self.selected_indices_]
        
        for i in range(len(X_selected)):
            try:
                for j in range(self.n_features):
                    self.fuzzy_sim.input[f'feature_{j}'] = X_selected[i, j]
                
                self.fuzzy_sim.compute()
                
                output_val = self.fuzzy_sim.output['genre']
                predicted_class = int(np.round(output_val))
                
                predicted_class = np.clip(predicted_class, 0, len(self.classes_) - 1)
                predictions.append(predicted_class)
                
            except Exception as e:
               
                predictions.append(0)
        
        return np.array(predictions)
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        model_data = {
            'n_features': self.n_features,
            'classes_': self.classes_,
            'centers_': self.centers_,
            'selected_indices_': self.selected_indices_,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Carrega um modelo salvo"""
        model_data = joblib.load(filepath)
        self.n_features = model_data['n_features']
        self.classes_ = model_data['classes_']
        self.centers_ = model_data['centers_']
        self.selected_indices_ = model_data['selected_indices_']
        self.feature_names = model_data['feature_names']