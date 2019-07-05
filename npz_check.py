import numpy as np
import cv2


def load_data_val(file):
    npzfile = np.load(file)
    val_data = npzfile["val_data"]
    val_label = npzfile["val_label"]
    return val_data, val_label

def load_data(file):
    npzfile = np.load(file)
    train_data = npzfile["train_data"]
    train_label = npzfile["train_label"]
    return train_data, train_label



if __name__ == '__main__':
    train_data, train_label = load_data("data_train1.npz")
    print(train_data.shape[0])
    for i in range(train_data.shape[0]):
        print(train_data[i])

    # val_data, val_label = load_data_val("data_val.npz")
    # print(val_data.shape)
    # for i in range(val_data.shape[0]):
    #     print(val_data[i])
