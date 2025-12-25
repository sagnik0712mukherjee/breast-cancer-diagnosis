import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

class ModelTrainer:
    def __init__(self, model_params):
        self.model_params = model_params
        self.best_models = {}
        self.scores = []

    def tune_models(self, inputs, target):
        """Performs RandomizedSearchCV for all models in model_params."""
        for model_name, mp in self.model_params.items():
            rscv_clf = RandomizedSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_iter=10)
            rscv_clf.fit(inputs, target.values.ravel())
            
            self.scores.append({
                'Model Name': model_name,
                'Best Score': rscv_clf.best_score_,
                'Best Parameter': rscv_clf.best_params_
            })
            self.best_models[model_name] = rscv_clf.best_estimator_
            
        return pd.DataFrame(self.scores, columns=['Model Name', 'Best Score', 'Best Parameter'])

    def train_final_model(self, model_class, params, inputs, target):
        """Trains a final model with specific parameters."""
        clf = model_class(**params)
        clf.fit(inputs, target.values.ravel())
        return clf
