# file to get all xlsx on folder, and merge on a single csv

import pandas as pd
import glob
import os


_folders = [
    'new_grouped_Log_2024-06-27',
    'new_grouped_Log_2024-06-28',
    'new_grouped_Log_2024-06-29',
]


def main():
    dfs = []
    for _folder in _folders:
        folders = glob.glob(_folder + '/*')
        print(folders)
        for folder in folders:
            excel_df = pd.read_csv(
                folder
            )
            print("games: #", len(excel_df))
            dfs.append(excel_df)

    data_df = pd.concat(dfs, ignore_index=True)

    print("-games: #", len(data_df), ".", len(dfs))

    data_df.to_csv('original_games.csv', index=False)


if __name__ == '__main__':
    main()
