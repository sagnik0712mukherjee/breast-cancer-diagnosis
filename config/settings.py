from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

MODEL_PARAMS = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [False, True],
            'copy_X': [False, True]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {
            'multi_class': ['auto', 'ovr'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C': [1, 2, 3, 4, 5, 10],
            'max_iter': [100, 800, 1000],
            'tol': [1e-3, 1e-4, 1e-5]
        }
    },
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [1, 2, 3, 4, 5, 10],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['auto', 'scale'],
            'decision_function_shape': ['ovo', 'ovr'],
            'tol': [1e-3, 1e-4, 1e-5]
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': [None, 'sqrt', 'log2']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 20, 50, 100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    },
    'DecisionTreeRegressor': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'splitter': ['best', 'random'],
            'max_features': [None, 'sqrt', 'log2']
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 20, 50, 100, 150, 200],
            'criterion': ['squared_error', 'absolute_error'],
            'max_features': [None, 'sqrt', 'log2']
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-09, 1e-10, 1e-11, 1e-12]
        }
    },
}
