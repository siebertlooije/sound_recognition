import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    with open('deel1transposed.csv', 'rU') as csvfile:
        datareader = csv.reader(csvfile)
        data = []
        for row in datareader:
            data.append(row)
        data = np.matrix(data)
        data = data.astype(np.float)

    # show_plot(data)

    return data


def derivation(data):
    row, col = data.shape

    deriv_data = data
    if row == 1:
        for x in range(1, col):
            deriv_data[x] = data[x] - data[x - 1]
    else:
        for x in range(col):
            for y in range(1, row):
                deriv_data[y, x] = data[y, x] - data[y - 1, x]

    return deriv_data


def activity_measurement(data):
    activities = []
    for x in range(0, len(rowSum2)):
        if data[x] >= data[int(0.5*len(data))]:
            activities.append(1)
        else:
            activities.append(0)

    return np.asarray(activities)


def show_plot(data):
    plt.plot(data)
    plt.show()

if __name__=="__main__":
    # do the thing

    raw_data = read_data()

    rowSum = np.sum(np.absolute(raw_data), 1)

    col, row = raw_data.shape

    rowSum = derivation(rowSum)

    deriv_data = derivation(np.absolute(raw_data))

    rowSum2 = np.absolute(np.sum(deriv_data, 1))

    activities = activity_measurement(rowSum)
    activities2 = activity_measurement(rowSum2)

    information_array = np.zeros(col)
    for x in range(0, col):
        information_array[x] = activities[x]

    sumactivities = []
    for x in range(0, len(information_array) - 10, 10):
        sumactivities.append(sum(information_array[x:x + 10]))

    show_plot(sumactivities)

