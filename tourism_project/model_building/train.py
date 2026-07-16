
# ==========================================================
# Visit With Us
# Wellness Tourism Package Purchase Prediction
#
# Production Training Pipeline
#
# Responsibilities
# 1. Load Dataset
# 2. Feature Engineering
# 3. Build Preprocessing Pipeline
# 4. Train XGBoost Model
# 5. Hyperparameter Tuning
# 6. MLflow Experiment Tracking
# 7. Evaluate Model
# 8. Save Pipeline
# 9. Upload Pipeline to Hugging Face
# ==========================================================

# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import warnings
import joblib

warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import xgboost as xgb

from huggingface_hub import HfApi

# Data Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

# Model Training
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

# Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "tourism_project/data/tourism.csv"

ARTIFACT_PATH = "tourism_project/artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_PATH,
    "model.pkl"
)

TARGET = "ProdTaken"

CLASSIFICATION_THRESHOLD = 0.45

REPO_ID = "ranjeetpatel29/Bank-Customer-Churn"

os.makedirs(
    ARTIFACT_PATH,
    exist_ok=True
)

# ==========================================================
# Hugging Face Configuration
# ==========================================================

api = HfApi(
    token=os.getenv("HF_TOKEN")
)

# ==========================================================
# Configure MLflow
# ==========================================================

def configure_mlflow():

    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment(
        "Tourism_Package_Prediction"
    )

    print("MLflow configured successfully.")

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset Shape : {df.shape}")

    return df

# ==========================================================
# Create Age Groups
# ==========================================================

def create_age_group(df):

    print("Creating Age Groups...")

    bins = [0, 25, 35, 45, 55, 100]

    labels = [
        "Young Adult",
        "Adult",
        "Middle Age",
        "Senior Adult",
        "Senior Citizen"
    ]

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    return df

# ==========================================================
# Split Dataset
# ==========================================================

def split_dataset(df):

    print("=" * 80)
    print("Splitting Dataset")
    print("=" * 80)

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training Records : {len(Xtrain)}")
    print(f"Testing Records  : {len(Xtest)}")

    return Xtrain, Xtest, ytrain, ytest
    # ==========================================================
# Create Preprocessing Pipeline
# ==========================================================

def create_preprocessor(Xtrain):

    print("=" * 80)
    print("Creating Preprocessing Pipeline")
    print("=" * 80)

    # Identify numeric and categorical columns
    numeric_features = Xtrain.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = Xtrain.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"Numeric Features     : {len(numeric_features)}")
    print(f"Categorical Features : {len(categorical_features)}")

    # Numeric preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )

        ]

    )

    return preprocessor


# ==========================================================
# Create XGBoost Model
# ==========================================================

def create_model(ytrain):

    class_weight = (
        ytrain.value_counts()[0]
        /
        ytrain.value_counts()[1]
    )

    print(f"Scale Positive Weight : {class_weight:.2f}")

    model = xgb.XGBClassifier(

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        scale_pos_weight=class_weight,

        tree_method="hist"

    )

    return model


# ==========================================================
# Create Training Pipeline
# ==========================================================

def create_pipeline(preprocessor, model):

    print("=" * 80)
    print("Creating Training Pipeline")
    print("=" * 80)

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )

    return pipeline


# ==========================================================
# Hyperparameter Grid
# ==========================================================

def create_param_grid():

    param_grid = {

        "model__n_estimators": [
            100,
            200
        ],

        "model__max_depth": [
            3,
            5
        ],

        "model__learning_rate": [
            0.01,
            0.05
        ],

        "model__subsample": [
            0.8,
            1.0
        ],

        "model__colsample_bytree": [
            0.8,
            1.0
        ],

        "model__min_child_weight": [
            1,
            3
        ]

    }

    return param_grid


# ==========================================================
# Create Grid Search
# ==========================================================

def create_grid_search(pipeline):

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42

    )

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=create_param_grid(),

        scoring="roc_auc",

        cv=cv,

        verbose=2,

        n_jobs=-1

    )

    return grid_search
    # ==========================================================
# Train Model
# ==========================================================

def train_model(grid_search, Xtrain, ytrain):

    print("=" * 80)
    print("Training Pipeline")
    print("=" * 80)

    with mlflow.start_run(run_name="XGBoost_Pipeline"):

        # Train complete pipeline
        grid_search.fit(
            Xtrain,
            ytrain
        )

        best_model = grid_search.best_estimator_

        best_params = grid_search.best_params_

        best_cv_score = grid_search.best_score_

        print("\nTraining Completed Successfully.")
        print(f"Best ROC-AUC Score : {best_cv_score:.4f}")

        # Log best parameters
        mlflow.log_params(best_params)

        # Log best cross validation score
        mlflow.log_metric(
            "Best_CV_ROC_AUC",
            best_cv_score
        )

        return (
            best_model,
            best_params,
            best_cv_score
        )


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(
    best_model,
    Xtrain,
    Xtest,
    ytrain,
    ytest
):

    print("=" * 80)
    print("Evaluating Model")
    print("=" * 80)

    # Training Prediction
    train_probability = best_model.predict_proba(
        Xtrain
    )[:,1]

    train_prediction = (
        train_probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    # Testing Prediction
    test_probability = best_model.predict_proba(
        Xtest
    )[:,1]

    test_prediction = (
        test_probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    metrics = {

        "Train Accuracy":
            accuracy_score(
                ytrain,
                train_prediction
            ),

        "Train Precision":
            precision_score(
                ytrain,
                train_prediction
            ),

        "Train Recall":
            recall_score(
                ytrain,
                train_prediction
            ),

        "Train F1":
            f1_score(
                ytrain,
                train_prediction
            ),

        "Train ROC AUC":
            roc_auc_score(
                ytrain,
                train_probability
            ),

        "Test Accuracy":
            accuracy_score(
                ytest,
                test_prediction
            ),

        "Test Precision":
            precision_score(
                ytest,
                test_prediction
            ),

        "Test Recall":
            recall_score(
                ytest,
                test_prediction
            ),

        "Test F1":
            f1_score(
                ytest,
                test_prediction
            ),

        "Test ROC AUC":
            roc_auc_score(
                ytest,
                test_probability
            )

    }

    print("\nModel Performance")
    print("=" * 80)

    for metric, value in metrics.items():

        print(f"{metric:<20}: {value:.4f}")

        mlflow.log_metric(
            metric.replace(" ", "_"),
            value
        )

    print("\nClassification Report")
    print("=" * 80)

    print(
        classification_report(
            ytest,
            test_prediction
        )
    )

    print("\nConfusion Matrix")
    print("=" * 80)

    print(
        confusion_matrix(
            ytest,
            test_prediction
        )
    )
    
# ==========================================================
# Save Model Pipeline
# ==========================================================

def save_model(best_model):

    print("=" * 80)
    print("Saving Model Pipeline")
    print("=" * 80)

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    mlflow.log_artifact(
        MODEL_PATH
    )

    print(f"Pipeline saved successfully at:\n{MODEL_PATH}")


# ==========================================================
# Upload Model to Hugging Face
# ==========================================================

def upload_model():

    print("=" * 80)
    print("Uploading Pipeline to Hugging Face")
    print("=" * 80)

    api.upload_file(

        path_or_fileobj=MODEL_PATH,

        path_in_repo="model.pkl",

        repo_id=REPO_ID,

        repo_type="dataset"

    )

    print("Pipeline uploaded successfully.")


# ==========================================================
# Main Function
# ==========================================================

def main():

    # Configure MLflow
    configure_mlflow()

    # Load Dataset
    df = load_dataset()

    # Feature Engineering
    df = create_age_group(df)

    # Train-Test Split
    Xtrain, Xtest, ytrain, ytest = split_dataset(df)

    # Create Preprocessor
    preprocessor = create_preprocessor(Xtrain)

    # Create XGBoost Model
    model = create_model(ytrain)

    # Create Training Pipeline
    pipeline = create_pipeline(
        preprocessor,
        model
    )

    # Create Grid Search
    grid_search = create_grid_search(
        pipeline
    )

    # Train Model
    best_model, best_params, best_cv_score = train_model(
        grid_search,
        Xtrain,
        ytrain
    )

    # Evaluate Model
    evaluate_model(
        best_model,
        Xtrain,
        Xtest,
        ytrain,
        ytest
    )

    # Save Pipeline
    save_model(
        best_model
    )

    # Upload Pipeline
    upload_model()

    print("\n")
    print("=" * 80)
    print("Training Pipeline Completed Successfully")
    print("=" * 80)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n")
        print("=" * 80)
        print("Training Failed")
        print("=" * 80)

        print(error)

        raise
