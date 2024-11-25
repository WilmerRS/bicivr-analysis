import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, SMOTENC
from scipy.stats import entropy
from collections import Counter


def get_original_incidents_dataset(
        save: bool = False
) -> pd.DataFrame:
    games = pd.read_csv('games.csv')

    columns = [
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
        "level",
        "real_risk",
    ]
    original_incidents_dataset = games[columns].copy()

    binary_columns = [
        "is_intersection",
        "is_two_way_street",
        "sidewalk_climbs",
        "running_red_light",
        "drive_opposite_direction",
        "bad_roundabout",
        "driving_between_lanes",
        "crossings_without_priority",
        "is_bike_infringement",
    ]

    for col in binary_columns:
        original_incidents_dataset[col] = original_incidents_dataset[col].apply(
            lambda x: 'yes' if x == 1 else 'no'
        )

    if save:
        original_incidents_dataset.to_csv(
            'thesis_datasets/original_incidents_dataset.csv', index=False
        )

    print(
        'Original incidents dataset shape:',
        original_incidents_dataset.shape,
    )

    return original_incidents_dataset


def get_original_collision_incidents_dataset(
        save: bool = False
) -> pd.DataFrame:
    games = pd.read_csv('games.csv')

    columns = [
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
        "bike_side_on_collision",
        "vehicle_side_on_collision",
        "in_collision_fault_of",
        "vehicle_speed_mean",
        "vehicle_speed_min",
        "vehicle_speed_max",
        "distance_mean",
        "distance_min",
        "distance_max",
        "bike_speed_mean",
        "bike_speed_min",
        "bike_speed_max",
        "level",
        "real_risk",
    ]
    original_collision_incidents_dataset = games[columns].copy()

    original_collision_incidents_dataset = original_collision_incidents_dataset[
        original_collision_incidents_dataset['real_risk'] == 'collision'
    ]

    binary_columns = [
        "is_intersection",
        "is_two_way_street",
        "sidewalk_climbs",
        "running_red_light",
        "drive_opposite_direction",
        "bad_roundabout",
        "driving_between_lanes",
        "crossings_without_priority",
        "is_bike_infringement",
    ]

    for col in binary_columns:
        original_collision_incidents_dataset[col] = original_collision_incidents_dataset[col].apply(
            lambda x: 'yes' if x == 1 else 'no'
        )

    if save:
        original_collision_incidents_dataset.to_csv(
            'thesis_datasets/original_collision_incidents_dataset.csv', index=False
        )

    print(
        'Original incidents dataset shape:',
        original_collision_incidents_dataset.shape,
    )

    return original_collision_incidents_dataset


def get_incidents_dataset_without_redundance(
        save: bool = False
) -> pd.DataFrame:
    original = pd.read_csv('thesis_datasets/original_incidents_dataset.csv')

    remove = [
        "vehicle_speed_mean",
        "vehicle_speed_min",
        "distance_mean",
        "distance_max",
        "bike_speed_mean",
        "bike_speed_min",
    ]

    incidents_dataset = original.drop(columns=remove)

    if save:
        incidents_dataset.to_csv(
            'thesis_datasets/incidents_dataset_without_redundance.csv',
            index=False
        )

    print(
        'Incidents dataset shape:',
        incidents_dataset.shape,
    )


def get_incidents_dataset_without_redundance_label_encoding(
        save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/incidents_dataset_without_redundance.csv'
    )

    label_encoder = LabelEncoder()

    categorical_columns = [
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
        "level",
        "real_risk",
    ]

    for col in categorical_columns:
        incidents_dataset[col] = label_encoder.fit_transform(
            incidents_dataset[col]
        )

    if save:
        incidents_dataset.to_csv(
            'thesis_datasets/incidents_dataset_without_redundance_label_encoding.csv',
            index=False
        )

    print(
        'Incidents dataset shape:',
        incidents_dataset.shape,
    )


def get_incidents_dataset_without_redundance_one_hot_encoding(
        save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/incidents_dataset_without_redundance.csv'
    )

    categorical_columns = [
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
        "level",
    ]

    oh_encoder = OneHotEncoder(
        sparse_output=False,
    )
    l_encoder = LabelEncoder()
    incidents_dataset['real_risk'] = l_encoder.fit_transform(
        incidents_dataset['real_risk']
    )

    encoded = oh_encoder.fit_transform(
        incidents_dataset[categorical_columns]
    )

    incidents_dataset = pd.concat(
        [
            incidents_dataset.drop(columns=categorical_columns),
            pd.DataFrame(
                encoded,
                columns=oh_encoder.get_feature_names_out(categorical_columns)
            ),
        ],
        axis=1,
    )

    if save:
        incidents_dataset.to_csv(
            'thesis_datasets/incidents_dataset_without_redundance_one_hot_encoding.csv',
            index=False
        )

    print(
        'Incidents dataset shape:',
        incidents_dataset.shape,
    )

# 80 - 20 split


def get_incidents_dataset_without_redundance_80_20_split(
        save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/incidents_dataset_without_redundance.csv'
    )

    X = incidents_dataset.drop(columns='real_risk')
    y = incidents_dataset['real_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    incidents_dataset_train = pd.concat(
        [X_train, y_train],
        axis=1
    )

    incidents_dataset_test = pd.concat(
        [X_test, y_test],
        axis=1
    )

    count_real_risk = incidents_dataset_train['real_risk'].value_counts()
    print('Train dataset distribution:', count_real_risk)
    count_real_risk = incidents_dataset_test['real_risk'].value_counts()
    print('Test dataset distribution:', count_real_risk)

    if save:
        incidents_dataset_train.to_csv(
            'thesis_datasets/80_20/incidents_dataset_without_redundance_train.csv',
            index=False
        )

        incidents_dataset_test.to_csv(
            'thesis_datasets/80_20/incidents_dataset_without_redundance_test.csv',
            index=False
        )

    print(
        'Incidents dataset shape:',
        incidents_dataset_train.shape,
        incidents_dataset_test.shape,
    )


def get_incidents_dataset_without_redundance_label_encoding_80_20_split(
        save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/incidents_dataset_without_redundance_label_encoding.csv'
    )

    X = incidents_dataset.drop(columns='real_risk')
    y = incidents_dataset['real_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    incidents_dataset_train = pd.concat(
        [X_train, y_train],
        axis=1
    )

    incidents_dataset_test = pd.concat(
        [X_test, y_test],
        axis=1
    )

    count_real_risk = incidents_dataset_train['real_risk'].value_counts()
    print('Train dataset distribution:', count_real_risk)
    count_real_risk = incidents_dataset_test['real_risk'].value_counts()
    print('Test dataset distribution:', count_real_risk)

    if save:
        incidents_dataset_train.to_csv(
            'thesis_datasets/80_20/incidents_dataset_without_redundance_label_encoding_train.csv',
            index=False
        )

        incidents_dataset_test.to_csv(
            'thesis_datasets/80_20/incidents_dataset_without_redundance_label_encoding_test.csv',
            index=False
        )

    print(
        'Incidents dataset shape:',
        incidents_dataset_train.shape,
        incidents_dataset_test.shape,
    )


def get_incidents_dataset_without_redundance_one_hot_encoding_80_20_split(
    save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/incidents_dataset_without_redundance_one_hot_encoding.csv'
    )

    X = incidents_dataset.drop(columns='real_risk')
    y = incidents_dataset['real_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    incidents_dataset_train = pd.concat(
        [X_train, y_train],
        axis=1
    )

    incidents_dataset_test = pd.concat(
        [X_test, y_test],
        axis=1
    )

    count_real_risk = incidents_dataset_train['real_risk'].value_counts()
    print('Train dataset distribution:', count_real_risk)
    count_real_risk = incidents_dataset_test['real_risk'].value_counts()
    print('Test dataset distribution:', count_real_risk)

    if save:
        incidents_dataset_train.to_csv(
            'thesis_datasets/80_20/incidents_dataset_without_redundance_one_hot_encoding_train.csv',
            index=False
        )

        incidents_dataset_test.to_csv(
            'thesis_datasets/80_20/incidents_dataset_without_redundance_one_hot_encoding_test.csv',
            index=False
        )

    print(
        'Incidents dataset shape:',
        incidents_dataset_train.shape,
        incidents_dataset_test.shape,
    )

# smote


def get_incidents_dataset_without_redundance_label_encoding_80_smote(
    save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/80_20_orange/incidents_dataset_without_redundance_label_encoding_train_orange.csv'
    )

    X = incidents_dataset.drop(columns='real_risk')
    y = incidents_dataset['real_risk']

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    incidents_dataset_smote = pd.concat(
        [X_smote, y_smote],
        axis=1
    )

    count_real_risk = incidents_dataset_smote['real_risk'].value_counts()
    print('SMOTE dataset distribution:', count_real_risk)

    if save:
        incidents_dataset_smote.to_csv(
            'thesis_datasets/80_smote_orange/incidents_dataset_without_redundance_label_encoding_80_smote_orange.csv',
            index=False
        )

    distribution = Counter(incidents_dataset_smote['real_risk'])
    _entropy = entropy(list(distribution.values()), base=2)

    print(
        'Incidents dataset shape (Label Encoding):',
        incidents_dataset_smote.shape,
        f'Entropy: {_entropy}'
    )


def get_incidents_dataset_without_redundance_one_hot_encoding_80_smote(
    save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/80_20_orange/incidents_dataset_without_redundance_one_hot_encoding_train_orange.csv'
    )

    X = incidents_dataset.drop(columns='real_risk')
    y = incidents_dataset['real_risk']

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    incidents_dataset_smote = pd.concat(
        [X_smote, y_smote],
        axis=1
    )

    count_real_risk = incidents_dataset_smote['real_risk'].value_counts()
    print('SMOTE dataset distribution:', count_real_risk)

    if save:
        incidents_dataset_smote.to_csv(
            'thesis_datasets/80_smote_orange/incidents_dataset_without_redundance_one_hot_encoding_80_smote_orange.csv',
            index=False
        )

    distribution = Counter(incidents_dataset_smote['real_risk'])
    _entropy = entropy(list(distribution.values()), base=2)

    print(
        'Incidents dataset shape (One Hot Encoding):',
        incidents_dataset_smote.shape,
        f'Entropy: {_entropy}'
    )


def get_incidents_dataset_without_redundance_80_smotenc(
        save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/80_20_orange/incidents_dataset_without_redundance_train_orange.csv'
    )
    print(incidents_dataset.head(2))

    X = incidents_dataset.drop(columns='real_risk')
    y = incidents_dataset['real_risk']

    categorical_columns = [
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
        # "is_bike_infringement",
        "level",
    ]

    smote = SMOTENC(
        random_state=42,
        categorical_features=categorical_columns,
    )
    X_smote, y_smote = smote.fit_resample(X, y)

    incidents_dataset_smote = pd.concat(
        [X_smote, y_smote],
        axis=1
    )

    count_real_risk = incidents_dataset_smote['real_risk'].value_counts()
    print('SMOTENC dataset distribution:', count_real_risk)

    if save:
        incidents_dataset_smote.to_csv(
            'thesis_datasets/80_smote_orange/incidents_dataset_without_redundance_80_smotenc_orange.csv',
            index=False
        )

    distribution = Counter(incidents_dataset_smote['real_risk'])
    _entropy = entropy(list(distribution.values()), base=2)

    print(
        'SMOTENC Incidents dataset shape (SMOTE):',
        incidents_dataset_smote.shape,
        f'Entropy: {_entropy}'
    )


def get_incidents_dataset_without_redundance_80_partial_smotenc(
        save: bool = False
) -> pd.DataFrame:
    incidents_dataset = pd.read_csv(
        'thesis_datasets/80_20_orange/incidents_dataset_without_redundance_train_orange.csv'
    )

    X = incidents_dataset.drop(columns='real_risk')
    y = incidents_dataset['real_risk']

    categorical_columns = [
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
        # "is_bike_infringement",
        "level",
    ]

    smote = SMOTENC(
        random_state=42,
        categorical_features=categorical_columns,
        sampling_strategy={
            'warning': 1345,
            'danger': int(503 * 1.5),
            'collision': 50 * 5,
        }
    )
    X_smote, y_smote = smote.fit_resample(X, y)

    incidents_dataset_smote = pd.concat(
        [X_smote, y_smote],
        axis=1
    )

    count_real_risk = incidents_dataset_smote['real_risk'].value_counts()
    print('SMOTENC dataset distribution:', count_real_risk)

    if save:
        incidents_dataset_smote.to_csv(
            'thesis_datasets/80_smote_orange/incidents_dataset_without_redundance_80_partial_smotenc_orange.csv',
            index=False
        )

    distribution = Counter(incidents_dataset_smote['real_risk'])
    _entropy = entropy(list(distribution.values()))
    print(
        'SMOTENC Incidents dataset shape:',
        incidents_dataset_smote.shape,
        f'Entropy: {_entropy}'
    )


def main():
    # get_original_incidents_dataset(save=True)
    # get_original_collision_incidents_dataset(save=True)
    # get_incidents_dataset_without_redundance(save=True)
    # get_incidents_dataset_without_redundance_label_encoding(save=True)
    # get_incidents_dataset_without_redundance_one_hot_encoding(save=True)

    # 80 - 20 split
    # get_incidents_dataset_without_redundance_80_20_split(save=True)
    # get_incidents_dataset_without_redundance_label_encoding_80_20_split(
    #     save=True
    # )
    # get_incidents_dataset_without_redundance_one_hot_encoding_80_20_split(
    #     save=True
    # )

    # smote
    get_incidents_dataset_without_redundance_label_encoding_80_smote(
        save=True
    )
    get_incidents_dataset_without_redundance_one_hot_encoding_80_smote(
        save=True
    )
    get_incidents_dataset_without_redundance_80_smotenc(save=True)
    get_incidents_dataset_without_redundance_80_partial_smotenc(
        save=True
    )


if __name__ == '__main__':
    main()
