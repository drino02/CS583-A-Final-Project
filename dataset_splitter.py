import splitfolders

if __name__ == "__main__":
    # Split each class folder into train, validation, and test sets
    splitfolders.ratio("Dataset/", output="dataset_split", seed=1337, ratio=(.65, 0.15, 0.2))