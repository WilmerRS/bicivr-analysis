import pandas as pd

def main():
    users = pd.read_csv('users.csv')

    # normalización de variables con respecto al total
    # total advertencias, total riesgos, total colisiones, total incidentes (logs)
    users['total advertencias'] = users['total advertencias'] / users['total incidentes (logs)']


if __name__ == '__main__':
    main()
