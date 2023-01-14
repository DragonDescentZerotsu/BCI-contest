import tensorflow as tf
import os
import numpy as np
import pickle as pkl


# from tensorflow.python.keras.models import load_model


def load_model_path(personID):
    dir_name_normal = 'Algorithm/model/normal/S0{}'.format(personID)
    # dir_name_EA = 'Algorithm/model/EA'
    model_list_normal = os.listdir(dir_name_normal)
    model_path_list = []
    for model in model_list_normal:
        model_path_list.append(os.path.join(dir_name_normal, model))
    # model_path_normal = os.path.join(dir_name_normal, model_list_normal[personID - 1])
    # model_list_EA = os.listdir(dir_name_EA)
    # model_path_EA = os.path.join(dir_name_EA, model_list_EA[personID - 1])
    return model_path_list


def recognize(data, personID):
    with open('Algorithm/model/EAs.pkl', 'rb') as fb:
        EAs = pkl.load(fb)
    EA = EAs[personID - 1]
    model_path_list = load_model_path(personID)
    model_fold_1 = tf.keras.models.load_model(model_path_list[0])
    # model_fold_2 = tf.keras.models.load_model(model_path_list[1])
    # model_fold_3 = tf.keras.models.load_model(model_path_list[2])
    # model_EA = tf.keras.models.load_model(model_path_EA)

    data = np.array(data)
    data_EA = []
    for i in range(len(data)):
        data_EA.append(np.dot(EA, data[i]))
    data = np.reshape(data, (data.shape[0], 20, 128, 1))
    data_EA = np.array(data_EA)
    data_EA = np.reshape(data_EA, (data_EA.shape[0], 20, 128, 1))

    if personID == 1:
        result_1 = model_fold_1.predict(data)
        # result_2 = model_fold_2.predict(data)
        # result_3 = model_fold_3.predict(data)
    elif personID == 2:
        result_1 = model_fold_1.predict(data)
        # result_2 = model_fold_2.predict(data_EA)
        # result_3 = model_fold_3.predict(data)
    elif personID == 3:
        result_1 = model_fold_1.predict(data)
        # result_2 = model_fold_2.predict(data_EA)
        # result_3 = model_fold_3.predict(data_EA)
    elif personID == 4:
        model_fold_2 = tf.keras.models.load_model(model_path_list[1])
        result_1 = model_fold_1.predict(data)
        result_2 = model_fold_2.predict(data)
        result_1 = result_1 * 0.5 + result_2 * 0.5
        # result_3 = model_fold_3.predict(data)
    elif personID == 5:
        result_1 = model_fold_1.predict(data)
        # result_2 = model_fold_2.predict(data_EA)
        # result_3 = model_fold_3.predict(data)
    # result_2 = model_EA.predict(data_EA)
    # weight = [[0.1, 0.7, 0.2],
    #           [0.65, 0.15, 0.2],
    #           [0.5, 0.1, 0.4],
    #           [0.4, 0.4, 0.2],
    #           [0.4, 0.3, 0.3]]
    # result_sum = weight[personID - 1][0] * result_1 + weight[personID - 1][1] * result_2 + weight[personID - 1][
    #     2] * result_3

    # try:

    vote = np.zeros((3,))
    for i in range(len(result_1)):
        temp = list(result_1[i])
        vote_number = temp.index(max(temp))
        if vote_number == 0:
            vote[0] = vote[0] + 1
        elif vote_number == 1:
            vote[1] = vote[1] + 1
        else:
            vote[2] = vote[2] + 1
    vote = list(vote)
    result = vote.index(max(vote)) + 201
    if vote[0] == vote[1] or vote[0] == vote[2] or vote[1] == vote[2]:
        result = np.where(result_1 == np.max(result_1))[1][0] + 201
    # except IndexError:
    #     print('IndexError')
    #     result = 203
    # else:
    print(result_1)
    # print(result_2)
    # print(result_3)
    # print(result_sum)
    print(vote)
    print(str(result) + ' / ' + str(personID))
    return result
