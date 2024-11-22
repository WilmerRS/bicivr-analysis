import pandas as pd


def main():
    games = pd.read_csv('games.csv')  # size = 1904
    original_games = pd.read_csv('original_games.csv')  # size = 2560

    # insert vehicle_position_x_min into games, filter by row_id
    # must be a search on original_games
    games['vehicle_position_x_min'] = games['row_id'].apply(
        lambda x: original_games[original_games['row_id']
                                 == x]['vehicle_position_x_min'].values[0]
    )
    games['vehicle_position_y_min'] = games['row_id'].apply(
        lambda y: original_games[original_games['row_id']
                                 == y]['vehicle_position_y_min'].values[0]
    )
    games['vehicle_position_z_min'] = games['row_id'].apply(
        lambda z: original_games[original_games['row_id']
                                 == z]['vehicle_position_z_min'].values[0]
    )

    games['vehicle_position_x_max'] = games['row_id'].apply(
        lambda x: original_games[original_games['row_id']
                                 == x]['vehicle_position_x_max'].values[0]
    )
    games['vehicle_position_y_max'] = games['row_id'].apply(
        lambda y: original_games[original_games['row_id']
                                 == y]['vehicle_position_y_max'].values[0]
    )
    games['vehicle_position_z_max'] = games['row_id'].apply(
        lambda z: original_games[original_games['row_id']
                                 == z]['vehicle_position_z_max'].values[0]
    )

    games['bike_position_x_min'] = games['row_id'].apply(
        lambda x: original_games[original_games['row_id']
                                 == x]['bike_position_x_min'].values[0]
    )
    games['bike_position_y_min'] = games['row_id'].apply(
        lambda y: original_games[original_games['row_id']
                                 == y]['bike_position_y_min'].values[0]
    )
    games['bike_position_z_min'] = games['row_id'].apply(
        lambda z: original_games[original_games['row_id']
                                 == z]['bike_position_z_min'].values[0]
    )

    games['bike_position_x_max'] = games['row_id'].apply(
        lambda x: original_games[original_games['row_id']
                                 == x]['bike_position_x_max'].values[0]
    )
    games['bike_position_y_max'] = games['row_id'].apply(
        lambda y: original_games[original_games['row_id']
                                 == y]['bike_position_y_max'].values[0]
    )
    games['bike_position_z_max'] = games['row_id'].apply(
        lambda z: original_games[original_games['row_id']
                                 == z]['bike_position_z_max'].values[0]
    )

    print(games[['vehicle_position_x_mean', 'vehicle_position_z_mean',
          'vehicle_position_x_min', 'vehicle_position_z_min']].head())

    games.to_csv('position_games.csv', index=False)


if __name__ == '__main__':
    main()
