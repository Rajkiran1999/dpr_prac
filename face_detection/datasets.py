import os
import glob
import wget
import tarfile

from absl import logging
import numpy as np

def download_and_prepare(name, dir):
    if name == "utk-faces":
        logging.info(f"Preparing dataset {name}...")
        # Check if data has been extracted and if not download extract it
        if os.path.isdir(os.path.join(dir, "UTKFace")):
            logging.info(f"Dataset {name} already extracted.")
        else:
            if not os.path.exists(os.path.join(dir, "UTKFace.tar.gz")):
                logging.info(f"UTKFace.tar.gz does not exist, please download it manually from https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE")
            else:
                logging.info(f"Extracting dataset {name}...")
                tar = tarfile.open(os.path.join(dir, "UTKFace.tar.gz"))
                tar.extractall(dir)
                tar.close()

        # Get a list of all files in dataset
        files = sorted(glob.glob(os.path.join(dir, "UTKFace", "*.jpg")))
        logging.debug(f"{len(files)} files read.")

        return files


    elif name == "sun2012":
        logging.info(f"Preparing dataset {name}...")
        # Check if data has been extracted and if not download extract it
        if os.path.isdir(os.path.join(dir, "SUN2012", "Images")):
            logging.info(f"Dataset {name} already extracted.")
        else:
            logging.info(f"Downloading dataset {name}...")
            url = "http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz"
            wget.download(url, dir)
            logging.info(f"Extracting dataset {name}...")
            tar = tarfile.open(os.path.join(dir, "SUN2012.tar.gz"))
            tar.extractall(dir)
            tar.close()

        # Get a list of all files in dataset
        files = sorted(glob.glob(os.path.join(dir, "SUN2012", "Images", "**", "*.jpg"), recursive=True))
        logging.debug(f"{len(files)} files read.")

        return files


    elif name == "fddb":
        logging.info(f"Preparing dataset {name}...")
        # Check if data has been extracted and if not download extract it
        if os.path.isdir(os.path.join(dir, "FDDB")) and os.path.isdir(os.path.join(dir, "FDDB", "FDDB-folds")):
            logging.info(f"Dataset {name} already extracted.")
        else:
            logging.info(f"Downloading dataset {name}...")
            url = "http://tamaraberg.com/faceDataset/originalPics.tar.gz"
            wget.download(url, dir)
            url = "http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz"
            wget.download(url, dir)
            logging.info(f"Extracting dataset {name}...")
            tar = tarfile.open(os.path.join(dir, "originalPics.tar.gz"))
            tar.extractall(os.path.join(dir, "FDDB"))
            tar.close()
            tar = tarfile.open(os.path.join(dir, "FDDB-folds.tgz"))
            tar.extractall(os.path.join(dir, "FDDB"))
            tar.close()

        # Get a list of all files in dataset
        files = sorted(glob.glob(os.path.join(dir, "FDDB", "FDDB-folds", "*ellipseList.txt"), recursive=True))
        labels = {}
        for file in files:
            with open(file, "r") as file_:
                state = "path"
                for line in file_:
                    line = line.rstrip()
                    if state == "path":
                        filepath = os.path.join(dir, "FDDB", line + ".jpg")
                        labels[filepath] = []
                        state = "number"
                    elif state == "number":
                        label_ns = int(line)
                        label_n = 0
                        state = "ellipses"
                    elif state == "ellipses":
                        major_axis_radius, minor_axis_radius, angle, center_x, center_y = line.split(" ")[:5]
                        major_axis_radius, minor_axis_radius, center_x, center_y = int(float(major_axis_radius)), \
                                                                                   int(float(minor_axis_radius)), \
                                                                                   int(float(center_x)), \
                                                                                   int(float(center_y))
                        angle = int(float(angle) / np.pi * 180) # convert angle from radial to degrees
                        labels[filepath].append([major_axis_radius, minor_axis_radius, angle, center_x, center_y])
                        label_n += 1
                        if label_n == label_ns:
                            state = "path"

        logging.debug(f"{len(labels.keys())} files read.")

        return list(labels.keys()), list(labels.values())

    else:
        raise ValueError
