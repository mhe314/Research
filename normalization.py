import numpy as np
import statistics as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pprint import pprint as pp


def std_dev(entry_data: dict) -> str:

    output_str = ''

    for key in entry_data.keys():
        p = st.pstdev(entry_data[key])
        output_str += f"Standard deviation of {key}: {round(p, 2)}\n"

    return output_str


def std_mean(entry_data: dict) -> str:

    output_str = ''

    for key in entry_data.keys():
        n = len(entry_data[key])
        mean = sum(entry_data[key]) / n
        output_str += f"Mean of {key}: {round(mean, 2)}\n"

    return output_str


def std_norm(entry_data: dict) -> str:

    data = []
    data_length = 0
    counter = 0

    for key in entry_data.keys():
        if counter == 0:
            data_length = len(entry_data[key])
            counter += 1
        else:
            if data_length != len(entry_data[key]):
                output_str = "ERROR: Number of data points\nmust be same for all categories\n"
                return output_str

        data.append(entry_data[key])

    data = np.array(data)

    pp(data)
    data.reshape(data.shape[1], data.shape[0])
    pp(data)
    scaler = StandardScaler()
    y = scaler.fit_transform(data)
    pp(y)
    output_str = f"Standard Normalized data:\n"

    for i in y:
        output_str += f"{i}\n"

    return output_str


def fs_min(entry_data: dict) -> str:

    output_str = ''

    for key in entry_data.keys():
        minimum = min(entry_data[key])
        output_str += f"Minimum of {key}: {round(minimum, 2)}\n"

    return output_str


def fs_max(entry_data: dict) -> str:

    output_str = ''

    for key in entry_data.keys():
        maximum = max(entry_data[key])
        output_str += f"Maximum of {key}: {round(maximum, 2)}\n"

    return output_str


def fs_norm(entry_data: dict) -> str:

    data = []
    data_length = 0
    counter = 0

    for key in entry_data.keys():
        if counter == 0:
            data_length = len(entry_data[key])
            counter += 1
        else:
            if data_length != len(entry_data[key]):
                output_str = "ERROR: Number of data points\nmust be same for all categories\n"
                return output_str

        data.append(entry_data[key])

    data = np.array(data)
    pp(data)

    data.reshape(data.shape[1], data.shape[0])
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(data)
    pp(y)
    output_str = f"Feature Scaled data:\n"

    for i in y:
        output_str += f"{i}\n"

    return output_str
