import numpy as np
from mne.decoding import CSP


def csp_yx(EEGSignals_x_train, EEGSignals_y_train):
    EEG_Channels = 20
    EEG_Trials = 280
    EEG_Classes = 2
    class_labels = [1, 2]

    # 计算协方差矩阵 cov1 cov2
    trialCov = np.zeros((EEG_Channels, EEG_Channels, EEG_Trials))
    for i in range(280):
        E = EEGSignals_x_train[:, :, i].T
        #     print(E.shape)
        EE = np.dot(E, E.T)
        trialCov[:, :, i] = EE / np.trace(EE)

    cov1 = np.zeros((EEG_Channels, EEG_Channels, 140))
    cov2 = np.zeros((EEG_Channels, EEG_Channels, 140))

    MatrixCov = np.zeros((3, 3, 2))
    k = 0
    s = 0
    for i in range(EEG_Channels):
        if EEGSignals_y_train[i] == 1:
            cov1[:, :, k] = trialCov[:, :, i]
            k = k + 1
        if EEGSignals_y_train[i] == 2:
            cov2[:, :, s] = trialCov[:, :, i]
            s = s + 1

    np.set_printoptions(precision=4)

    # 计算平均协方差矩阵之和 covTotal
    cov1 = np.mean(cov1, 2)
    cov2 = np.mean(cov2, 2)
    covTotal = cov1 + cov2
    print(covTotal)

    # 计算公共特征向量矩阵和特征值
    [Dt, Uc] = np.linalg.eigh(covTotal)
    # 降序排序
    Uc = Uc[:, Dt.argsort()[::-1]]
    Dt = sorted(Dt, reverse=True)

    # 矩阵白化
    Dt = np.diag(Dt)
    Dt = np.sqrt(1. / Dt)

    # 去 inf值
    c = np.isinf(Dt)
    Dt[c] = 0
    print(Dt)
    P = np.dot(Dt, Uc.T)

    #     print(P)

    # 将P作用于 cov1 cov2
    transformedCov1 = np.dot(np.dot(P, cov1), P.T)
    transformedCov2 = np.dot(np.dot(P, cov2), P.T)  # S1 = P*R1*P.T

    # 将 transformedCov1 按主分量分解得到公共特征向量矩阵 B

    [D1, U1] = np.linalg.eig(transformedCov1)
    # 降序
    U1 = U1[:, D1.argsort()]
    D1 = sorted(D1, reverse=True)

    #     print(U1)

    [D2, U2] = np.linalg.eig(transformedCov2)
    # 升序
    U2 = U2[:, D2.argsort()]
    D2 = np.sort(D2)
    #     print(U2)

    # 计算投影矩阵
    CSPMatrix = np.dot(U1.T, P)

    return CSPMatrix


