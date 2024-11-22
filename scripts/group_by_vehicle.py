import pandas as pd
import glob

FOLDER = 'Log_2024-06-27'
FILE_NAME = 'n115894_2024-06-27.csv'


folders = [
    'Log_2024-06-27',
    'Log_2024-06-28',
    'Log_2024-06-29',
]


def read_data(file_name):
    return pd.read_csv(file_name)


def group_by_unique_events(folder, grouped_folder, filename):
    data = read_data(f"{folder}/{filename}")

    data['current_date_time'] = pd.to_datetime(
        data['currentDateTime'],
        format='%Y-%m-%d %H:%M:%S'
    )
    data['time_formatted'] = pd.to_datetime(data['time'], format='%H:%M:%S.%f')

    last_seconds = 1
    data['rowId'] = (
        data['currentDateTime'].str.slice(0, -1 * last_seconds)
        # + data['currentDateTime'].str.slice(18, 20).apply(
        #     lambda x: '50' if int(x) < 5 else '59'
        # )
        + '_vID'
        + data['vehicleId'].astype(str)
    )
    data_grouped = data.groupby('rowId').agg(
        row_id=('rowId', 'first'),
        current_date_time_min=('current_date_time', 'min'),
        current_date_time_mean=('current_date_time', 'mean'),
        current_date_time_max=('current_date_time', 'max'),
        time_mean=('time_formatted', 'mean'),
        time_min=('time_formatted', 'min'),
        time_max=('time_formatted', 'max'),
        user_id=('userId', 'first'),
        risk=('risk', 'first'),
        vehicle_id=('vehicleId', 'first'),
        vehicle_name=('vehicleName', 'first'),
        vehicle_position_x_mean=('vehiclePositionX', 'mean'),
        vehicle_position_y_mean=('vehiclePositionY', 'mean'),
        vehicle_position_z_mean=('vehiclePositionZ', 'mean'),
        vehicle_position_x_min=('vehiclePositionX', 'min'),
        vehicle_position_y_min=('vehiclePositionY', 'min'),
        vehicle_position_z_min=('vehiclePositionZ', 'min'),
        vehicle_position_x_max=('vehiclePositionX', 'max'),
        vehicle_position_y_max=('vehiclePositionY', 'max'),
        vehicle_position_z_max=('vehiclePositionZ', 'max'),
        bike_position_x_mean=('bikePositionX', 'mean'),
        bike_position_y_mean=('bikePositionY', 'mean'),
        bike_position_z_mean=('bikePositionZ', 'mean'),
        bike_position_x_min=('bikePositionX', 'min'),
        bike_position_y_min=('bikePositionY', 'min'),
        bike_position_z_min=('bikePositionZ', 'min'),
        bike_position_x_max=('bikePositionX', 'max'),
        bike_position_y_max=('bikePositionY', 'max'),
        bike_position_z_max=('bikePositionZ', 'max'),
        vehicle_speed_mean=('vehicleSpeed', 'mean'),
        vehicle_speed_min=('vehicleSpeed', 'min'),
        vehicle_speed_max=('vehicleSpeed', 'max'),
        distance_mean=('distance', 'mean'),
        distance_min=('distance', 'min'),
        distance_max=('distance', 'max'),
        bike_speed_mean=('bikeSpeed', 'mean'),
        bike_speed_min=('bikeSpeed', 'min'),
        bike_speed_max=('bikeSpeed', 'max'),
    )
    data_grouped['time_min'] = data_grouped['time_min'].dt.time
    data_grouped['time_max'] = data_grouped['time_max'].dt.time
    data_grouped['time_mean'] = data_grouped['time_mean'].dt.time

    danger_radius = 2
    data_grouped['real_risk'] = data_grouped['distance_min'].apply(
        lambda x:
        'collision'
        if x <= 0
        else (
            'danger'
            if x < danger_radius
            else 'warning'
        )
    )

    data_grouped['distance_min'] = data_grouped['distance_min'].apply(
        lambda x: 0 if x < 0 else x
    )
    data_grouped['distance_mean'] = data_grouped['distance_mean'].apply(
        lambda x: 0 if x < 0 else x
    )
    data_grouped['distance_max'] = data_grouped['distance_max'].apply(
        lambda x: 0 if x < 0 else x
    )

    data_grouped['lanes'] = ''
    data_grouped['is_intersection'] = ''
    data_grouped['is_two_way_street'] = ''
    data_grouped['bike_maneuver'] = ''
    data_grouped['is_bike_infringement'] = ''

    data_grouped['sidewalk_climbs'] = ''
    data_grouped['running_red_light'] = ''
    data_grouped['drive_opposite_direction'] = ''
    data_grouped['bad_roundabout'] = ''
    data_grouped['driving_between_lanes'] = ''
    data_grouped['crossings_without_priority'] = ''

    data_grouped = data_grouped[[
        'row_id',
        'current_date_time_min',
        'current_date_time_mean',
        'current_date_time_max',
        'time_mean',
        'time_min',
        'time_max',
        'user_id',
        'lanes',
        'is_intersection',
        'is_two_way_street',
        'bike_maneuver',
        'sidewalk_climbs',
        'running_red_light',
        'drive_opposite_direction',
        'bad_roundabout',
        'driving_between_lanes',
        'crossings_without_priority',
        'is_bike_infringement',
        'real_risk',
        'risk',
        'vehicle_id',
        'vehicle_name',
        'vehicle_position_x_mean',
        'vehicle_position_y_mean',
        'vehicle_position_z_mean',
        'vehicle_position_x_min',
        'vehicle_position_y_min',
        'vehicle_position_z_min',
        'vehicle_position_x_max',
        'vehicle_position_y_max',
        'vehicle_position_z_max',
        'bike_position_x_mean',
        'bike_position_y_mean',
        'bike_position_z_mean',
        'bike_position_x_min',
        'bike_position_y_min',
        'bike_position_z_min',
        'bike_position_x_max',
        'bike_position_y_max',
        'bike_position_z_max',
        'vehicle_speed_mean',
        'vehicle_speed_min',
        'vehicle_speed_max',
        'distance_mean',
        'distance_min',
        'distance_max',
        'bike_speed_mean',
        'bike_speed_min',
        'bike_speed_max'
    ]]
    data_grouped.to_csv(f"{grouped_folder}/{filename}", index=False)


def main():
    print("Starting group_by_vehicle.py")

    for folder in folders:
        files = glob.glob(f"{folder}/*.csv")
        for file in files:
            group_by_unique_events(
                folder,
                f"new_grouped_{folder}",
                file.split('\\')[-1]
            )
            print(f"Grouped {file}")

    print("Finished group_by_vehicle.py")


if __name__ == '__main__':
    main()
