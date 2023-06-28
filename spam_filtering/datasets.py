import os
import wget
import zipfile

import pandas as pd
from absl import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def download_and_prepare(name, path):
    if name == "sms_spam":
        logging.info(f"Preparing dataset {name}...")
        # Check if data has been extracted and if not download extract it
        if (os.path.exists(os.path.join(path, "SMSSpamCollection"))):
            logging.info(f"Dataset {name} already extracted.")
        else:
            logging.info(f"Downloading dataset {name}...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
            wget.download(url, path)
            logging.info(f"Extracting dataset {name}...")
            with zipfile.ZipFile(os.path.join(path, "smsspamcollection.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(path, "SMSSpamCollection"))

        # Read dataset with pandas
        dataset = pd.read_csv(os.path.join(path, "SMSSpamCollection", "SMSSpamCollection"), delimiter="\t", encoding="latin-1", header=None)
        logging.debug(f"{len(dataset)} entries read.")

        # Preprocessing
        dataset[2] = dataset[0].map({'ham': 0, 'spam': 1})
        X, y = dataset[1].values, dataset[2].values
        vect = CountVectorizer(stop_words='english')
        vect.fit(X)
        X = vect.transform(X).toarray()

        # Train-test-split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test

    else:
        raise ValueError

