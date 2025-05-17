import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance, DMatrix, train as xgb_train
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from collections import Counter
import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning)

def make_serializable(obj):
    """
    Recursively convert NumPy data types in the input object to native Python types.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def train_and_predict_with_xgb():
    dataset_names = []
    X_trains = []
    y_trains = []
    X_tests = []
    auc_scores = []
    dataset_auc = {}
    dataset_metrics = {}
    dataset_best_params = {}

    low_auc_datasets = [
        'Dataset_4', 'Dataset_11', 'Dataset_24', 'Dataset_31',
        'Dataset_32', 'Dataset_43', 'Dataset_47'
    ]
    
    data_path = "./Competition_data/Competition_data"
    
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                X_train = pd.read_csv(os.path.join(folder_path, "X_train.csv"), header=0)
                y_train = pd.read_csv(os.path.join(folder_path, "y_train.csv"), header=0)
                X_test = pd.read_csv(os.path.join(folder_path, "X_test.csv"), header=0)
            except FileNotFoundError as e:
                print(f"Error reading files for {folder_name}: {e}")
                continue
            dataset_names.append(folder_name)
            X_trains.append(X_train)
            y_trains.append(y_train.squeeze().values)  # Ensure y_train is 1D NumPy array
            X_tests.append(X_test)

    for i in range(len(dataset_names)):
        try:
            print(f"Processing {dataset_names[i]}...")

            # Load data
            X_train = X_trains[i]
            y_train = y_trains[i]
            X_test = X_tests[i]

            # Handle missing values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())

            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train-test split
            tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            # Compute scale_pos_weight to handle imbalance
            counter = Counter(tmp_y_train)
            majority_class = max(counter, key=counter.get)
            minority_class = min(counter, key=counter.get)
            scale_pos_weight = counter[majority_class] / counter[minority_class] if counter[minority_class] != 0 else 1

            # Hyperparameter selection
            if dataset_names[i] in low_auc_datasets:
                param_dist = {
                    'n_estimators': np.arange(100, 501, 100),
                    'max_depth': np.arange(3, 11, 1),
                    'learning_rate': np.linspace(0.01, 0.2, 20),
                    'subsample': np.linspace(0.6, 1.0, 5),
                    'colsample_bytree': np.linspace(0.6, 1.0, 5),
                    'gamma': [0, 1, 5],
                    'min_child_weight': [1, 5, 10],
                    'scale_pos_weight': [1, scale_pos_weight]
                }
            else:
                param_dist = {
                    'n_estimators': [200],
                    'max_depth': [6],
                    'learning_rate': [0.05],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'gamma': [0],
                    'min_child_weight': [1],
                    'scale_pos_weight': [1]
                }

            # Model definition
            xgb_model = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            )

            # Hyperparameter tuning using RandomizedSearchCV
            randomized_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_dist,
                n_iter=50,
                scoring='roc_auc',
                cv=3,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            randomized_search.fit(tmp_X_train, tmp_y_train)
            best_params = randomized_search.best_params_
            dataset_best_params[dataset_names[i]] = best_params

            # Prepare DMatrices for training with early stopping
            dtrain = DMatrix(tmp_X_train, label=tmp_y_train)
            dtest = DMatrix(tmp_X_test, label=tmp_y_test)

            # Define final model parameters with best hyperparameters
            params = randomized_search.best_params_
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42
            })

            # Train the model using xgb.train with early stopping
            evals = [(dtrain, 'train'), (dtest, 'eval')]
            best_model = xgb_train(
                params,
                dtrain,
                num_boost_round=500,
                evals=evals,
                early_stopping_rounds=20,
                verbose_eval=False
            )

            # Predict and evaluate
            tmp_y_prob = best_model.predict(dtest)
            auc = roc_auc_score(tmp_y_test, tmp_y_prob)

            tmp_y_pred = (tmp_y_prob >= 0.5).astype(int)
            precision = precision_score(tmp_y_test, tmp_y_pred)
            recall = recall_score(tmp_y_test, tmp_y_pred)
            f1 = f1_score(tmp_y_test, tmp_y_pred)

            auc_scores.append(auc)
            dataset_auc[dataset_names[i]] = auc
            dataset_metrics[dataset_names[i]] = {
                'AUC': auc,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }

            print(f"Dataset {dataset_names[i]} - AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

            # Predict for the test set
            dtest_full = DMatrix(X_test_scaled)
            y_predict_proba = best_model.predict(dtest_full)

            # Save the prediction results
            output_df = pd.DataFrame({
                "y_predict_proba": y_predict_proba
            })
            output_csv_path = os.path.join(data_path, dataset_names[i], "y_predict.csv")
            output_df.to_csv(output_csv_path, index=False)

            # Plot feature importance
            plt.figure(figsize=(10, 8))
            plot_importance(best_model, max_num_features=10, importance_type='weight')
            plt.title(f'Feature Importance for {dataset_names[i]}')
            plt.tight_layout()
            plt.savefig(os.path.join(data_path, dataset_names[i], "feature_importance.png"))
            plt.close()

        except Exception as e:
            print(f"An error occurred while processing {dataset_names[i]}: {e}")

    # Save overall results
    average_auc = np.mean(auc_scores)
    print(f"\nAverage XGBoost AUC Across All Datasets: {average_auc:.4f}")

    metrics_df = pd.DataFrame(dataset_metrics).T
    metrics_csv_path = os.path.join(data_path, "dataset_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=True)

    serializable_best_params = make_serializable(dataset_best_params)

    hyperparams_json_path = os.path.join(data_path, "best_hyperparameters.json")
    with open(hyperparams_json_path, 'w') as f:
        json.dump(serializable_best_params, f, indent=4)

    # Plot AUC scores per dataset
    plt.figure(figsize=(15, 6))
    datasets = list(dataset_auc.keys())
    auc_values = list(dataset_auc.values())
    sns.barplot(x=datasets, y=auc_values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Dataset')
    plt.ylabel('AUC Score')
    plt.title('AUC Score per Dataset')
    plt.tight_layout()
    auc_plot_path = os.path.join(data_path, "auc_scores_per_dataset.png")
    plt.savefig(auc_plot_path)
    plt.close()

train_and_predict_with_xgb()
