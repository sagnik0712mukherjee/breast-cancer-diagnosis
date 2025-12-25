from src.data_loader import load_data, preprocess_data, get_train_test_split
from src.trainer import ModelTrainer
from src.utils import plot_correlation_heatmap, plot_confusion_matrix, print_model_scores
from config.settings import MODEL_PARAMS
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. Load Data
    data_path = 'src/data/Breast_cancer_data.csv'
    df = load_data(data_path)
    
    # 2. Visualize Correlation (Optional)
    # plot_correlation_heatmap(df)
    
    # 3. Preprocess Data
    inputs, target = preprocess_data(df)
    
    # 4. Tune Models
    trainer = ModelTrainer(MODEL_PARAMS)
    results_df = trainer.tune_models(inputs, target)
    print_model_scores(results_df)
    
    # 5. Train Final Model (RandomForest as per original script)
    # best params could be extracted from results_df, but here we follow original script's choice
    final_params = {
        'class_weight': 'balanced_subsample',
        'criterion': 'entropy',
        'max_features': 'log2',
        'n_estimators': 200
    }
    
    X_train, X_test, y_train, y_test = get_train_test_split(inputs, target)
    
    final_clf = trainer.train_final_model(RandomForestClassifier, final_params, X_train, y_train)
    
    # 6. Evaluate
    y_pred = final_clf.predict(X_test)
    score = final_clf.score(X_test, y_test)
    print(f"\nFinal Model Score on Test Set: {score:.4f}")
    
    # 7. Plot Confusion Matrix
    # plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()
