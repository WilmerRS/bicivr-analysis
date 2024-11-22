import pandas as pd
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
    "level",
    "real_risk"
]

target_column = "real_risk"


def compare_similarities(games, games_train_smote_nc):
    """
    Counts the number of exact matches in games_train_smote_nc and games.
    """
    games = games[independent_columns]
    games_train_smote_nc = games_train_smote_nc[independent_columns]

    games = games[games['real_risk'] == 'danger']
    games_train_smote_nc = games_train_smote_nc[games_train_smote_nc['real_risk'] == 'danger']

    print(f"[👋] total games_train_smote_nc: {len(games_train_smote_nc)}")
    print(f"[👋] total games: {len(games)}")

    uniques_games_train_smote_nc = games_train_smote_nc.copy().drop_duplicates(keep=False)
    print(f"[👋] uniques_games_train_smote_nc: {
          len(uniques_games_train_smote_nc)}")
    original_uniques = games.copy().drop_duplicates(keep=False)
    print(f"[👋] original_uniques: {len(original_uniques)}")

    concatenated = pd.concat([games, games_train_smote_nc])

    duplicates = concatenated[concatenated.duplicated(keep=False)]

    print(f"duplicates: {len(duplicates)}")

    uniques = concatenated.drop_duplicates(keep=False)
    print(f"uniques: {len(uniques)}")
    print(f"news: {len(uniques) + len(games_train_smote_nc)}")


def generateId(x: pd.Series):
    print(type(x))
    # conver to string
    x = x.astype(str)
    # join all values
    print("[👋]", '__'.join(x))
    return '__'.join(x)


def find_repeated(original: pd.DataFrame, smote_nc: pd.DataFrame):
    original['line_id'] = original[independent_columns].astype(str).apply(
        lambda x: '__'.join(x),
        axis=1
    )
    smote_nc['line_id'] = smote_nc[independent_columns].astype(str).apply(
        lambda x: '__'.join(x),
        axis=1
    )

    original = original[original['real_risk'] == 'warning']
    smote_nc = smote_nc[smote_nc['real_risk'] == 'warning']

    print(f"[🥳] original: {len(original)}")
    print(f"[🥳] smoted: {len(smote_nc)}")

    # count how many smoted values exist in original, take as reference line_id
    exist_on_original = smote_nc['line_id'].isin(original['line_id'])
    print(f"[🥳] exist_on_original: {exist_on_original.sum()}")


def main():
    games = pd.read_csv('games.csv')

    games_train_smote_nc = pd.read_csv('games_train__smote_nc.csv')

    compare_similarities(games, games_train_smote_nc)

    find_repeated(games, games_train_smote_nc)


if __name__ == '__main__':
    main()
