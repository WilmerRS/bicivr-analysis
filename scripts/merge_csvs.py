# file to get all xlsx on folder, and merge on a single csv

import pandas as pd
import glob
import os

FOLDER = os.path.join('games')
FILE_NAME = 'games.csv'


def main():
    folders = glob.glob(FOLDER + '/*')
    print(folders)
    dfs = []
    for folder in folders:
        excel_df = pd.read_excel(
            folder
        )
        print("games: #", len(excel_df))
        dfs.append(excel_df)

    data_df = pd.concat(dfs, ignore_index=True)

    print("-games: #", len(data_df), ".", len(dfs))

    data_df.to_csv(FILE_NAME, index=False)


if __name__ == '__main__':
    main()
