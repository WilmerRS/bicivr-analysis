import pandas as pd
import itertools


def main():
    users = pd.read_csv('users.csv')
    games = pd.read_csv('games.csv')

    # Ver las primeras filas del dataset
    # print(list(games.columns))
    # 'row_id'
    # 'current_date_time_min'
    # 'current_date_time_mean'
    # 'current_date_time_max'
    # 'time_mean'
    # 'time_min'
    # 'time_max'
    # 'user_id'
    # 'lanes'
    # 'is_intersection'
    # 'is_two_way_street'
    # 'bike_maneuver'
    # 'vehicle_maneuver'
    # 'maneuver_direction'
    # 'sidewalk_climbs'
    # 'running_red_light'
    # 'drive_opposite_direction'
    # 'bad_roundabout'
    # 'driving_between_lanes'
    # 'crossings_without_priority'
    # 'is_bike_infringement'
    # 'real_risk'
    # 'risk'
    # 'bike_side_on_collision'
    # 'vehicle_side_on_collision'
    # 'in_collision_fault_of'
    # 'vehicle_id'
    # 'vehicle_name'
    # 'vehicle_position_x_mean'
    # 'vehicle_position_y_mean'
    # 'vehicle_position_z_mean'
    # 'bike_position_x_mean'
    # 'bike_position_y_mean'
    # 'bike_position_z_mean'
    # 'vehicle_speed_mean'
    # 'vehicle_speed_min'
    # 'vehicle_speed_max'
    # 'distance_mean'
    # 'distance_min'
    # 'distance_max'
    # 'bike_speed_mean'
    # 'bike_speed_min'
    # 'bike_speed_max'
    # 'level'

    # print(games.head())

    # # Ver el resumen de la estructura del dataset
    # print(games.info())

    # # Ver un resumen estadístico rápido de las variables numéricas
    # print(games.describe())
    # summary = games.describe(include='all')
    # print(summary)

    # get_frecuency(games)
    # get_total_infringements(games)
    # get_velocity_means(games)
    # get_collisions_analisys(games)
    # get_infriengements(games)
    # get_collison_infringements(games)
    # get_infriengements_level(games)
    # get_collision_infringements(games)
    # get_faults_frequency(games)
    # get_faults_combinations(games)
    # get_infrastructure_frequency(games)
    # relation_maneuver_with_risk(games)
    relation_level_with_risk(games)


def get_frecuency(df):
    categorical_columns = ['sidewalk_climbs', 'running_red_light', 'drive_opposite_direction',
                           'bad_roundabout', 'driving_between_lanes', 'crossings_without_priority', 'is_bike_infringement']
    total = 0
    for column in categorical_columns:
        frequency_table = df[column].value_counts()
        relative_frequency = df[column].value_counts(normalize=True) * 100

        # Crear un DataFrame para la tabla de frecuencias
        frequency_df = pd.DataFrame({
            'Frecuencia Absoluta': frequency_table,
            'Frecuencia Relativa (%)': relative_frequency
        })

        # Mostrar la tabla de frecuencia
        print(f'Tabla de frecuencias para {column}:')
        print(frequency_df)
        print('\n')

    print("--total", total, "--")


def get_total_infringements(df):
    # create a bolean colums of is_bike_infringement
    # to create, compute if least one of the columns is True
    df['is_bike_infringement'] = df[['sidewalk_climbs', 'running_red_light', 'drive_opposite_direction',
                                     'bad_roundabout', 'driving_between_lanes', 'crossings_without_priority']].any(axis=1)
    total_infringements = df['is_bike_infringement'].sum()
    print(f'Total de infracciones: {total_infringements}')


def get_velocity_means(df):
    # calcular velocidad media por nivel
    mean_velocity = df.groupby('real_risk')['vehicle_speed_max'].mean()
    print("vehicles mean", mean_velocity)
    # min_velocity = df.groupby('level')['vehicle_speed_min'].min()
    # print("vehicles min", min_velocity)
    # max_velocity = df.groupby('level')['vehicle_speed_max'].max()
    # print("vehicles max", max_velocity)

    bike_mean_velocity = df.groupby('real_risk')['bike_speed_max'].mean()
    print("bikes mean", bike_mean_velocity)
    # bike_min_velocity = df.groupby('level')['bike_speed_min'].min()
    # print("bikes min", bike_min_velocity)
    # bike_max_velocity = df.groupby('level')['bike_speed_max'].max()
    # print("bikes max", bike_max_velocity)


def get_collisions_analisys(df):
    # create a bolean colums of is_bike_infringement
    # to create, compute if least one of the columns is True
    df['is_bike_infringement'] = df[['sidewalk_climbs', 'running_red_light', 'drive_opposite_direction',
                                     'bad_roundabout', 'driving_between_lanes', 'crossings_without_priority']].any(axis=1)

    # count collisions with infractions. group by real_risk
    collisions = df.groupby('real_risk')['is_bike_infringement'].count()
    print("[👋] collisions", collisions)

    only_bike_infringements = df[df['is_bike_infringement'] == True].groupby(
        'real_risk')['is_bike_infringement'].count()
    print("[👋] only bike infringements", only_bike_infringements)


def get_infriengements(df):
    # create a bolean colums of is_bike_infringement
    # to create, compute if least one of the columns is True
    df['infringements'] = df[['sidewalk_climbs', 'running_red_light', 'drive_opposite_direction',
                              'bad_roundabout', 'driving_between_lanes', 'crossings_without_priority']].sum(axis=1)

    # count collisions with infractions. group by real_risk
    collisions = df.groupby('real_risk')['infringements'].mean()
    print("[👋] collisions", collisions)

    # solo donde hay real_rosk = collision
    get_frecuency(
        df[df['real_risk'] == 'collision']
    )


def get_infriengements_level(df):
    # create a bolean colums of is_bike_infringement
    # to create, compute if least one of the columns is True
    df['infringements'] = df[['sidewalk_climbs', 'running_red_light', 'drive_opposite_direction',
                              'bad_roundabout', 'driving_between_lanes', 'crossings_without_priority']].sum(axis=1)

    # count collisions with infractions. group by real_risk
    collisions = df.groupby('level')['infringements'].mean()
    print("[👋] get_infriengements_level", collisions)

    # solo donde hay level = hard
    get_frecuency(
        df[df['level'] == 'hard']
    )


def get_collision_infringements(df):
    # calcular cuantas infracciones se cometieron, agrupado por riesgo, cuando convinamos más de una infracción
    columns = ['sidewalk_climbs', 'running_red_light', 'drive_opposite_direction',
               'bad_roundabout', 'driving_between_lanes', 'crossings_without_priority']
    # create combinations of columns
    combinations = []
    for i in range(1, len(columns)+1):
        combinations += list(itertools.combinations(columns, i))

    df['result'] = df[columns].sum(axis=1)

    # Crear una columna que contenga la combinación que generó ese resultado
    df['combinacion'] = df.apply(
        lambda row: ''.join([col[0] for col in columns if row[col]]),
        axis=1
    )

    # get first 5 rows
    g = df[df['real_risk'] == 'collision'].groupby('combinacion')[
        'result'].count()
    print(g.sort_values(ascending=False))
    print((g / g.sum() * 100).sort_values(ascending=False))


def get_faults_frequency(df):
    only_collisions = df[df['real_risk'] == 'collision']
    cols = ['bike_side_on_collision', 'vehicle_side_on_collision',
            'in_collision_fault_of']

    for col in cols:
        frequency_table = only_collisions[col].value_counts()
        relative_frequency = only_collisions[col].value_counts(
            normalize=True) * 100

        # Crear un DataFrame para la tabla de frecuencias
        frequency_df = pd.DataFrame({
            'Frecuencia Absoluta': frequency_table,
            'Frecuencia Relativa (%)': relative_frequency
        })

        # Mostrar la tabla de frecuencia
        print(f'[✅] Tabla de frecuencias para {col}:')
        print(frequency_df)
        print('\n')


def get_faults_combinations(df):
    only_collisions = df[df['real_risk'] == 'collision']
    only_collisions['situations'] = 'bike_' + only_collisions['bike_side_on_collision'] + \
        '__car_' + only_collisions['vehicle_side_on_collision']
    frequency_table = only_collisions['situations'].value_counts()
    relative_frequency = only_collisions['situations'].value_counts(
        normalize=True) * 100

    frequency_df = pd.DataFrame({
        'Frecuencia Absoluta': frequency_table,
        'Frecuencia Relativa (%)': relative_frequency
    })

    print(f'[✅] Tabla de frecuencias para combinaciones de situaciones:')
    print(frequency_df)

    only_collisions['situations_faults'] = 'bike_' + only_collisions['bike_side_on_collision'] + \
        '__car_' + only_collisions['vehicle_side_on_collision'] + \
        '__fault_' + only_collisions['in_collision_fault_of']
    frequency_table = only_collisions['situations_faults'].value_counts()
    relative_frequency = only_collisions['situations_faults'].value_counts(
        normalize=True) * 100

    frequency_df = pd.DataFrame({
        'Frecuencia Absoluta': frequency_table,
        'Frecuencia Relativa (%)': relative_frequency
    })

    print(f'[✅] Tabla de frecuencias para combinaciones de situaciones y faltas:')
    print(frequency_df)


def get_infrastructure_frequency(df):
    only_collisions = df[df['real_risk'] == 'collision']
    cols = ['lanes', 'is_intersection', 'is_two_way_street']
    for col in cols:
        frequency_table = only_collisions[col].value_counts()
        relative_frequency = only_collisions[col].value_counts(
            normalize=True) * 100

        # Crear un DataFrame para la tabla de frecuencias
        frequency_df = pd.DataFrame({
            'Frecuencia Absoluta': frequency_table,
            'Frecuencia Relativa (%)': relative_frequency
        })

        # Mostrar la tabla de frecuencia
        print(f'[✅] Tabla de frecuencias para {col}:')
        print(frequency_df)
        print('\n')


def relation_maneuver_with_risk(games):
    risks = games['real_risk'].unique()
    for risk in risks:
        print(f'\n[✅] Analizando el riesgo: {risk}')
        only_this_risk = games[games['real_risk'] == risk]
        cols = ['bike_maneuver', 'vehicle_maneuver', 'maneuver_direction']
        for col in cols:
            frequency_table = only_this_risk[col].value_counts()
            relative_frequency = only_this_risk[col].value_counts(
                normalize=True) * 100

            # Crear un DataFrame para la tabla de frecuencias
            frequency_df = pd.DataFrame({
                'Frecuencia Absoluta': frequency_table,
                'Frecuencia Relativa (%)': relative_frequency
            })

            # Mostrar la tabla de frecuencia
            print(f' -- Tabla de frecuencias para {col}:')
            print(frequency_df)
            print('\n')


def relation_level_with_risk(games):
    risks = games['real_risk'].unique()
    for risk in risks:
        print(f'\n[✅] Analizando el riesgo: {risk}')
        only_this_risk = games[games['real_risk'] == risk]
        levels = ['level']
        for col in levels:
            frequency_table = only_this_risk[col].value_counts()
            relative_frequency = only_this_risk[col].value_counts(
                normalize=True) * 100

            # Crear un DataFrame para la tabla de frecuencias
            frequency_df = pd.DataFrame({
                'Frecuencia Absoluta': frequency_table,
                'Frecuencia Relativa (%)': relative_frequency
            })

            # Mostrar la tabla de frecuencia
            print(f' -- Tabla de frecuencias para {col}:')
            print(frequency_df)
            print('\n')


if __name__ == '__main__':
    main()
