import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

independent_columns = [
    "lanes",
    "is_intersection",
    "is_two_way_street",
    "bike_maneuver",
    "vehicle_maneuver",
    "maneuver_direction",
    "sidewalk_climbs",
    # "running_red_light",
    "drive_opposite_direction",
    "bad_roundabout",
    "driving_between_lanes",
    "crossings_without_priority",
    # "is_bike_infringement",
    # "vehicle_speed_mean",
    # "vehicle_speed_min",
    "vehicle_speed_max",
    # "distance_mean",
    # "distance_min",
    # "distance_max",
    # "bike_speed_mean",
    # "bike_speed_min",
    "bike_speed_max",
    "level"
]
categorical_columns = [
    "is_intersection",
    "is_two_way_street",
    "bike_maneuver",
    "vehicle_maneuver",
    "maneuver_direction",
    "sidewalk_climbs",
    # "running_red_light",
    "drive_opposite_direction",
    "bad_roundabout",
    "driving_between_lanes",
    "crossings_without_priority",
    # "is_bike_infringement",
    "level",
]


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets.

    Parameters:
    X: Features.
    y: Target.
    test_size: Proportion of dataset to include in the test split.
    random_state: Seed for reproducibility.

    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    smote_nc = SMOTENC(
        categorical_features=[
            X.columns.get_loc(col)
            for col in categorical_columns
        ],
        random_state=42
    )

    X_train_resampled, y_train_resampled = smote_nc.fit_resample(
        X_train,
        y_train
    )

    ohe = OneHotEncoder(
        sparse_output=False,
        drop='first',
        handle_unknown='ignore'
    )

    X_train_resampled_encoded = pd.DataFrame(
        ohe.fit_transform(X_train_resampled[categorical_columns]),
        columns=ohe.get_feature_names_out(categorical_columns),
        index=X_train_resampled.index
    )
    X_test_encoded = pd.DataFrame(
        ohe.transform(X_test[categorical_columns]),
        columns=ohe.get_feature_names_out(categorical_columns),
        index=X_test.index
    )

    # Add non-categorical columns back to the dataset
    X_train_resampled_encoded = pd.concat(
        [
            X_train_resampled.drop(columns=categorical_columns),
            X_train_resampled_encoded
        ],
        axis=1
    )
    X_test_encoded = pd.concat(
        [
            X_test.drop(columns=categorical_columns),
            X_test_encoded
        ],
        axis=1
    )

    label_mapping = {
        'collision': 2,
        'danger': 1,
        'warning': 0
    }
    y_train_resampled = y_train_resampled.map(label_mapping)
    y_test = y_test.map(label_mapping)

    print(f"Training Target Distribution: {y_train_resampled[0:5]}")
    print(f"Training Target Distribution: {y_test[0:5]}")
    print(f"Training Target Distribution: {X_train[0:5]}")

    return X_train_resampled_encoded, X_test_encoded, y_train_resampled, y_test


def build_xgboost_model():
    """
    Define the XGBoost model and hyperparameter grid for tuning.

    Returns:
    model: XGBoost classifier model.
    param_grid: Dictionary of hyperparameters for tuning.
    """
    model = xgb.XGBClassifier()

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.5],
        'min_child_weight': [1, 5, 10]
    }

    return model, param_grid


def hyperparameter_tuning(X_train, y_train, model, param_grid, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV and cross-validation.

    Parameters:
    X_train: Training features.
    y_train: Training target.
    model: XGBoost model.
    param_grid: Dictionary of hyperparameters for tuning.
    cv: Number of cross-validation folds.

    Returns:
    best_model: Model with the best hyperparameters.
    best_params: Dictionary of the best hyperparameters.
    """
    scorer = make_scorer(f1_score, average='weighted')
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_train, y_train, cv=5):
    """
    Evaluate the best model using cross-validation and F1 score.

    Parameters:
    model: The best XGBoost model.
    X_train: Training features.
    y_train: Training target.
    cv: Number of cross-validation folds.

    Returns:
    cv_scores: List of cross-validation scores (F1).
    mean_score: Mean F1 score across folds.
    """
    scorer = make_scorer(f1_score, average='weighted')
    cv_scores = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv)

    mean_score = cv_scores.mean()

    return cv_scores, mean_score


def run_pipeline(X, y):
    """
    Run the full pipeline for hyperparameter tuning and evaluation.

    Parameters:
    X: Features.
    y: Target.

    Returns:
    best_model: The best tuned XGBoost model.
    best_params: The best hyperparameters.
    cv_scores: Cross-validation scores for F1.
    mean_score: Mean F1 score.
    """
    # Step 1: Prepare Data
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Step 2: Build Model
    model, param_grid = build_xgboost_model()

    # Step 3: Perform Hyperparameter Tuning
    best_model, best_params = hyperparameter_tuning(
        X_train,
        y_train,
        model,
        param_grid
    )

    # Step 4: Evaluate Best Model
    cv_scores, mean_score = evaluate_model(best_model, X_train, y_train)

    print(f"Best Hyperparameters: {best_params}")
    print(f"Cross-Validation F1 Scores: {cv_scores}")
    print(f"Mean F1 Score: {mean_score:.4f}")

    return best_model, best_params, cv_scores, mean_score


def main():
    games = pd.read_csv('games.csv')

    X = games[independent_columns]
    y = games['real_risk']

    best_model, best_params, cv_scores, mean_f1 = run_pipeline(X, y)

    print("Training best model on full dataset...")
    print(f"Test F1 Score: {best_model}, {cv_scores}, {mean_f1}")
    print(f"Best params {best_params}")


if __name__ == '__main__':
    main()
