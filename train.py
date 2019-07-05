from keras import models, layers
import numpy as np
import npz_check as nc
import matplotlib.pyplot as plt

train_data, train_label = nc.load_data("data_train1.npz")
test_data, test_label = nc.load_data_val("data_val1.npz")
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
train_label = train_label * np.array((0.01, 0.01))
test_label = test_label * np.array((0.01, 0.01))

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def train(epoch_num):
    model = build_model()
    model.fit(train_data, train_label, epochs=epoch_num, batch_size=256, verbose=1)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_label)
    print(test_mse_score, test_mae_score)
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def test(loaded_model, X_test):
    X_test -= mean
    X_test /= std
    # print(X_test)
    ans = loaded_model.predict(X_test)
    return ans


def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

if __name__ == '__main__':
    train(epoch_num=75)

    # k = 3
    # num_val_samples = len(train_data) // k
    # num_epochs = 200
    # all_mae_histories = []
    #
    # for i in range(k):
    #     print('processing_data #', i)
    #     val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    #     val_target = train_label[i * num_val_samples: (i+1) * num_val_samples]
    #
    #     partial_train_data = np.concatenate(
    #         [train_data[:i * num_val_samples],
    #         train_data[(i+1) * num_val_samples:]],
    #         axis=0)
    #     partial_train_target = np.concatenate(
    #         [train_label[:i * num_val_samples],
    #          train_label[(i + 1) * num_val_samples:]],
    #         axis=0)
    #
    #     model = build_model()
    #     history = model.fit(partial_train_data, partial_train_target, validation_data=(val_data, val_target),
    #                         epochs=num_epochs, batch_size=128, verbose=1)
    #     mae_history = history.history['val_mean_absolute_error']
    #     all_mae_histories.append(mae_history)
    #
    # # print(all_mae_histories)
    # average_mae_history = [
    #     np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    # average_mae_history = smooth_curve(average_mae_history[:])
    # plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation MAE')
    # plt.show()
