import pandas as pd


def _get_sum_real_risk(risk, user_id, games):
    return games[games['user_id'] == user_id]['real_risk'].apply(
        lambda x: 1 if x == risk else 0
    ).sum()


def main():
    games = pd.read_csv('games.csv')
    users = pd.read_csv('users.csv')

    users['total advertencias'] = users['Usuario'].apply(
        lambda x: _get_sum_real_risk('warning', x, games)
    )
    users['total riesgos'] = users['Usuario'].apply(
        lambda x: _get_sum_real_risk('danger', x, games)
    )
    users['total colisiones'] = users['Usuario'].apply(
        lambda x: _get_sum_real_risk('collision', x, games)
    )
    users['total incidentes (logs)'] = users['Usuario'].apply(
        lambda x: games[games['user_id'] == x].shape[0]
    )

    users.to_csv('users.csv', index=False)


if __name__ == '__main__':
    main()
