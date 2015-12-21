import matplotlib.pyplot as plt
import time


def read_features(file_name):
    features = list()

    f = open(file_name)
    for line in f.readlines():
        feature_list = line.split(" ")
        feature = list()
        for item in feature_list:
            if item != '\n':
                feature.append(float(item))
        features.append(feature)

    f.close()
    return features
