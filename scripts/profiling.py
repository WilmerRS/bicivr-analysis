import pandas as pd
from ydata_profiling import ProfileReport


def main():
    games = pd.read_csv('games.csv')
    profile = ProfileReport(games, title="Profiling Report")

    profile.to_file("profiling_report.html")


if __name__ == '__main__':
    main()
