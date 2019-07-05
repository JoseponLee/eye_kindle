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

def Merge(data1, data2):
    merge = []
    for i in range(2):
        merge.append(np.vstack((data1[i], data2[i])))
    return merge


data = load_data("data_train.npz")
data1 = load_data("data_train5.npz")
# data = load_data("data_train3.npz")
# data2 = load_data("data_train4.npz")
# data_val = load_data_val("data_val.npz")
# data_val2 = load_data_val("data_val2.npz")

merge_train = Merge(data, data1)
# merge_train = Merge(merge_train, data2)
# merge_train = Merge(merge_train, data3)
# merge_train = Merge(merge_train, data4)
# merge_train = Merge(merge_train, data5)
# merge_val = Merge(data_val, data_val2)


np.savez('data_train_all.npz', train_data=np.array(merge_train[0]), train_label=np.array(merge_train[1]))