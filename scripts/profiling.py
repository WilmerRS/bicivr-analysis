import pandas as pd
from ydata_profiling import ProfileReport


def main():
    games = pd.read_csv('games.csv')
    profile = ProfileReport(
        games,
        title="Profiling Report",
        explorative=True,
        sensitive=True,
        correlations={
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
    )
    profile.to_file("profiling_report.html")


if __name__ == '__main__':
    main()
