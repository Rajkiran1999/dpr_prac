import os
import ssl
import wget
import zipfile

import numpy as np
import pandas as pd


def download_and_prepare(name, path):
    if name == "movielens-small":
        print(f"Preparing dataset {name}...")
        # Check if data has been extracted and if not download extract it
        if (os.path.exists(os.path.join(path, "ml-latest-small"))):
            print(f"Dataset {name} already extracted.")
        else:
            print(f"Downloading dataset {name}...")
            ssl._create_default_https_context = ssl._create_unverified_context
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            wget.download(url, path)
            print(f"Extracting dataset {name}...")
            with zipfile.ZipFile(os.path.join(path, "ml-latest-small.zip"), 'r') as zip_ref:
                zip_ref.extractall(path)

        # Read dataset with pandas
        ratings = pd.read_csv(os.path.join(path, 'ml-latest-small', 'ratings.csv'))
        print(f"{len(ratings)} entries read.")
        r_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        return np.array(r_matrix) # for performance reasons we only take every 2nd element along each axis

    else:
        raise ValueError

