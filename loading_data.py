"""Loading data sets"""
import pandas as pd


def add_rul_1(df):
    """

    :param df: raw data frame
    :return: data frame labeled with targets
    """
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (piece-wise Linear)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]

    result_frame["RUL"] = remaining_useful_life
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def load_FD001(cut):
    """

    :param cut: upper limit for target RULs
    :return: grouped data per sample
    """
    # load data FD001.py
    # define filepath to read data
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors

    train.drop(labels=drop_labels, axis=1, inplace=True)
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())  # min-max normalization
    # data_norm = (data-data.mean())/data.std()  # standard normalization (optional)
    train_norm = pd.concat([title, data_norm], axis=1)
    train_norm = add_rul_1(train_norm)
    # as in piece-wise linear function, there is an upper limit for target RUL,
    # however, experimental results shows this goes even better without it:
    # train_norm['RUL'].clip(upper=cut, inplace=True)
    group = train_norm.groupby(by="unit_nr")

    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    test_norm = pd.concat([title, data_norm], axis=1)
    group_test = test_norm.groupby(by="unit_nr")

    return group, group_test, y_test
