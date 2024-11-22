import pandas as pd
import itertools


def main():
    users = pd.read_csv('users.csv')

    # describe_dataset(users)
    # relation_infractions_by_level(users)
    # relation_infractions_detail_by_level(users)
    # relation_age_with_total_infractions(users)
    # relation_age_with_total_incidents(users)
    relation_sex_with_performance_user(users)


def describe_dataset(users: pd.DataFrame):
    for column in users.columns:
        print(f"[👋] {column}")

        frequency_table = users[column].value_counts()
        relative_frequency = users[column].value_counts(normalize=True) * 100
        print(
            pd.DataFrame({
                'Frecuencia Absoluta': frequency_table,
                'Frecuencia Relativa (%)': relative_frequency
            })
        )


def relation_infractions_by_level(users):
    level_key = "Nivel jugado"
    total_infractions_key = "total infracciones"

    levels = users[level_key].unique()
    for level in levels:
        print(f"[👋] analyzing {level}")
        users_level = users[users[level_key] == level]

        mean_infractions = users_level[total_infractions_key].mean()
        max_infractions = users_level[total_infractions_key].max()

        print(f"Mean infractions: {mean_infractions}")
        print(f"Max infractions: {max_infractions}")


def relation_infractions_detail_by_level(users):
    level_key = "Nivel jugado"
    infractions_key = [
        "Subidas a andén por calle",
        "Pasarse semaforo",
        "conducir sentido contrario por calle",
        "entrar mal a rotonda",
        "conducir por entre carriles por calle",
        "Cruces sin prioridad",
    ]

    levels = users[level_key].unique()
    for level in levels:
        print(f"[👋] analyzing {level}")
        for infraction in infractions_key:
            users_level = users[users[level_key] == level]

            mean_infractions = users_level[infraction].mean()
            max_infractions = users_level[infraction].max()

            print(f"Mean infractions [{infraction}]:  {mean_infractions:.2f}")
            print(f"Max  infractions [{infraction}]:  {max_infractions:.2f}")


def relation_age_with_total_infractions(users: pd.DataFrame):
    age_key = "Edad"
    total_infractions_key = "total infracciones"

    # Group by age range, with 4 years of difference.
    ranges = {
        '18-21': 0,
        '22-25': 0,
        '26-29': 0,
        '30-33': 0,
    }
    grouped_by_range = list(ranges.keys())

    users_ages = users[age_key].apply(
        lambda x:
            '18-21' if 18 >= x and x <= 21 else
            '22-25' if 22 >= x and x <= 25 else
            '26-29' if 26 >= x and x <= 29 else
            '30-33' if 30 >= x and x <= 33 else 0
    )

    for age in grouped_by_range:
        # print(f"[👋] analyzing {age}")
        users_level = users[users_ages == age]

        mean_age = users_level[total_infractions_key].mean()

        print(f"Age: {age} => Mean infractions: {mean_age}")
        # print(f"Max  infractions: {max_age}")


def relation_age_with_total_incidents(users: pd.DataFrame):
    age_key = "Edad"
    total_incidents_key = "total incidentes (logs)"

    # Group by age range, with 4 years of difference.
    ranges = {
        '18-21': 0,
        '22-25': 0,
        '26-29': 0,
        '30-33': 0,
    }
    grouped_by_range = list(ranges.keys())

    users_ages = users[age_key].apply(
        lambda x:
            '18-21' if 18 >= x and x <= 21 else
            '22-25' if 22 >= x and x <= 25 else
            '26-29' if 26 >= x and x <= 29 else
            '30-33' if 30 >= x and x <= 33 else 0
    )

    for age in grouped_by_range:
        # print(f"[👋] analyzing {age}")
        users_level = users[users_ages == age]

        mean_age = users_level[total_incidents_key].mean()

        print(f"Age: {age} => Mean incidents: {mean_age}")


def relation_sex_with_performance_user(users: pd.DataFrame):
    p_key = "Sexo"
    total_incidents_key = "total incidentes (logs)"
    total_collisions = "total colisiones"
    total_infractions_key = "total infracciones"

    # Group by age range, with 4 years of difference.
    sexs = users[p_key].unique()

    for sex in sexs:
        filtered_users = users[users[p_key] == sex]

        mean_incidents = filtered_users[total_incidents_key].mean()
        mean_collisions = filtered_users[total_collisions].mean()
        mean_infractions = filtered_users[total_infractions_key].mean()

        print(f"sex: {sex} => Mean incidents: {mean_incidents:.2f}")
        print(f"sex: {sex} => Mean collisions: {mean_collisions:.2f}")
        print(f"sex: {sex} => Mean infractions: {mean_infractions:.2f}\n")


if __name__ == '__main__':
    main()
