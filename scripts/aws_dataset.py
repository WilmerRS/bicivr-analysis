import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

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
    "vehicle_speed_max",
    "distance_min",
    "bike_speed_max",
    "level"
]


def main():
    games = pd.read_csv('games.csv')

    x = games[columns]
    y = games['real_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # join in a train dataset, and test dataset
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    # save
    train.to_csv('aws_train.csv', index=False)
    test.to_csv('aws_test.csv', index=False)

    print(f"[INFO] Train dataset: {Counter(y_train)}")
    print(f"[INFO] Test dataset: {Counter(y_test)}")

    all = pd.concat([train, test], axis=0)
    all.to_csv('aws_all.csv', index=False)
    

if __name__ == '__main__':
    main()
