import pandas as pd


def _get_level(user_id, users):
    level = users[users['Usuario'] == user_id]['Nivel jugado'].values[0]

    levels = {
        'Fácil': 'easy',
        'Normal': 'normal',
        'Difícil': 'hard',
    }

    return levels[level]


def main():
    games = pd.read_csv('games.csv')
    users = pd.read_csv('users.csv')
    games['level'] = games['user_id'].apply(
        lambda x: _get_level(x, users)
    )

    print(games['level'].head(2))

    games.to_csv('games.csv', index=False)


if __name__ == '__main__':
    main()
