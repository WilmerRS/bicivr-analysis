import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def _missing_analysis(df):
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        print("No missing values")
    else:
        print(missing_values)


def main():
    games = pd.read_csv('games.csv')

    print(f"distribución original: {Counter(games['real_risk'])}")

    independent_columns = [
        "lanes",
        "is_intersection",
        "is_two_way_street",
        "bike_maneuver",
        "vehicle_maneuver",
        "maneuver_direction",
        "sidewalk_climbs",
        "running_red_light",
        "drive_opposite_direction",
        "bad_roundabout",
        "driving_between_lanes",
        "crossings_without_priority",
        "is_bike_infringement",
        "vehicle_speed_mean",
        "vehicle_speed_min",
        "vehicle_speed_max",
        "distance_mean",
        "distance_min",
        "distance_max",
        "bike_speed_mean",
        "bike_speed_min",
        "bike_speed_max",
        "level"
    ]
    categorical_columns = [
        # "lanes",
        "is_intersection",
        "is_two_way_street",
        "bike_maneuver",
        "vehicle_maneuver",
        "maneuver_direction",
        "sidewalk_climbs",
        "running_red_light",
        "drive_opposite_direction",
        "bad_roundabout",
        "driving_between_lanes",
        "crossings_without_priority",
        "is_bike_infringement",
        "level"
    ]

    # _save_ohe_dataset(games, independent_columns, categorical_columns)
    # _save_le_dataset(games, independent_columns, categorical_columns)
    _save_categorical_smote_dataset(
        games, independent_columns, categorical_columns)
    # _save_categorical_smoteenn_dataset(
    #     games, independent_columns, categorical_columns)


def _save_ohe_dataset(games, independent_columns, categorical_columns):
    X = games[independent_columns]
    y = games['real_risk']

    label_mapping = {
        'collision': 2,
        'danger': 1,
        'warning': 0
    }

    y_encoded = y.map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(), categorical_columns),
        ],
        remainder='passthrough',
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_transformed,
        y_train
    )

    columns = preprocessor.get_feature_names_out()
    games_train_df = pd.DataFrame(
        X_train_resampled,
        columns=columns
    )
    games_train_df['real_risk'] = y_train_resampled
    games_train_df.to_csv('games_train__le.csv', index=False)

    games_test_df = pd.DataFrame(
        X_test_transformed,
        columns=columns,
        index=X_test.index
    )
    games_test_df['real_risk'] = y_test
    games_test_df.to_csv('games_test__le.csv', index=False)

    print(f"distribución después de oversampling: {
          Counter(y_train_resampled)}")


def _save_le_dataset(games, independent_columns, categorical_columns):
    X = games[independent_columns]
    y = games['real_risk']

    label_mapping = {
        'collision': 2,
        'danger': 1,
        'warning': 0
    }

    y_encoded = y.map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    for column in categorical_columns:
        label_encoder = LabelEncoder()
        X_train[column] = label_encoder.fit_transform(X_train[column])
        X_test[column] = label_encoder.transform(X_test[column])

    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    games_train_df = pd.concat(
        [X_train_resampled, y_train_resampled],
        axis=1
    )

    games_train_df.to_csv('games_train__le.csv', index=False)

    games_test_df = pd.concat(
        [X_test, y_test],
        axis=1
    )

    games_test_df.to_csv('games_test__le.csv', index=False)

    print(f"distribución después de oversampling: {
        Counter(y_train_resampled)}")


def _save_categorical_smote_dataset(games, independent_columns, categorical_columns):
    print(len(games))
    games['binary_risk'] = games['distance_min'].apply(
        lambda x: 'collision' if x <= 0.2 else 'danger'
    )
    # remove rows when distance_min > 5
    games = games[games['distance_min'] <= 2]
    print(games['distance_min'].min())
    print(games['distance_min'].mean())
    print(games['distance_min'].max())
    print(len(games))
    print("[games]", games['binary_risk'].count())

    X = games[independent_columns]
    y = games['binary_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    smote_nc = SMOTENC(
        categorical_features=[
            X.columns.get_loc(col)
            for col in categorical_columns
        ],
        random_state=42,
        # sampling_strategy={
        #     'collision': 624,
        #     'danger': 624,
        #     'warning': 624
        # }
        sampling_strategy={
            'danger': 379,
            'collision': 100
        }
    )

    X_train_resampled, y_train_resampled = smote_nc.fit_resample(
        X_train,
        y_train
    )

    games_train_df = pd.concat(
        [X_train_resampled, y_train_resampled],
        axis=1
    )
    games_test_df = pd.concat(
        [X_test, y_test],
        axis=1
    )

    games_train_df.to_csv('games_train__smote_nc__new_risk_danger_only_smote.csv', index=False)
    games_test_df.to_csv('games_test__smote_nc__new_risk_danger_only_smote.csv', index=False)

    print(f"[🍏] o_y_train: {Counter(y_train_resampled)}")
    print(f"[🍏] y_train: {Counter(y_train)}")
    print(f"[🍏] y_test:  {Counter(y_test)}")
    print(f"[🍏] y:       {Counter(y)}")


def _save_categorical_smoteenn_dataset(games: pd.DataFrame, independent_columns, categorical_columns):
    X: pd.DataFrame = games[independent_columns]
    y = games['real_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    le = {}
    for column in categorical_columns:
        label_encoder = LabelEncoder()
        le[column] = label_encoder
        X_train[column] = label_encoder.fit_transform(X_train[column])

    smote_enn = SMOTEENN(
        random_state=42,
    )

    X_train_resampled, y_train_resampled = smote_enn.fit_resample(
        X_train,
        y_train
    )

    for column in categorical_columns:
        label_encoder = le[column]
        X_train_resampled[column] = label_encoder.inverse_transform(
            X_train_resampled[column]
        )

    games_train_df = pd.concat(
        [X_train_resampled, y_train_resampled],
        axis=1
    )
    games_train_df.to_csv('games_train__smote_nc_enn.csv', index=False)
    games_test_df = pd.concat(
        [X_test, y_test],
        axis=1
    )
    games_test_df.to_csv('games_test__smote_nc_enn.csv', index=False)

    print(f"distribución después de oversampling: {
        Counter(y_train_resampled)}")


if __name__ == '__main__':
    main()
