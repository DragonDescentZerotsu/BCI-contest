from Algorithm.Interface.AlgorithmInterface import AlgorithmInterface
from Algorithm.Interface.Model.ReportModel import ReportModel
import numpy as np
import numpy.linalg as la
# from EEGModels import EEGNet
from Algorithm.recognize import recognize
from scipy import signal


# import time


class AlgorithmImplementMI(AlgorithmInterface):
    # 类属性：范式名称
    PARADIGMNAME = 'MI'

    def __init__(self):
        super().__init__()
        # 定义采样率，题目文件中给出
        samp_rate = 250
        # 选择导联编号
        self.select_channel = list(range(1, 60))
        self.select_channel = [i - 1 for i in self.select_channel]
        # 想象开始的trigger（由于240作为trial开始被占用，这里使用系统预留trigger:249）
        self.trial_stimulate_mask_trig = 249
        # trial结束trigger
        self.trial_end_trig = 241
        # 计算时间
        cal_time = 4
        # 计算长度
        self.cal_len = 128 + 50
        # 预处理滤波器设置
        self.filterB, self.filterA = self.__get_pre_filter(samp_rate)
        # 初始化方法
        # self.method = EEGNet()

    def run(self):
        # 是否停止标签
        end_flag = False
        # 是否进入计算模式标签
        cal_flag = False
        trial_count = 0
        sum = np.zeros((20, 20))
        record_ID = 0
        while not end_flag:
            data_model = self.task.get_data()
            if record_ID != data_model.subject_id:
                trial_count = 0
                sum = np.zeros((20, 20))
                record_ID = data_model.subject_id
            if not cal_flag:
                # 非计算模式，则进行事件检测
                cal_flag = self.__idle_proc(data_model)
            else:
                # 计算模式，则进行处理
                # start_time = time.time()
                cal_flag, result, sum, trial_count = self.__cal_proc(data_model, sum, trial_count)
                # 如果有结果，则进行报告
                if result is not None:
                    report_model = ReportModel()
                    report_model.result = result
                    self.task.report(report_model)
                    # 清空缓存
                    self.__clear_cache()
                # end_time = time.time()
                # print('Time: ' + str((end_time-start_time) * 1000) + 'ms')

            end_flag = data_model.finish_flag

    def __idle_proc(self, data_model):  # proc=procedure
        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导
        trigger = data[-1, :]  # 这表示最后一行所有数据就是trigger的导联
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]  # np.where的返回值是两层嵌套，[0]去掉一层
        # 脑电数据
        eeg_data = data[0: -1, :]  # 含左不含右，相当于取全部脑电数据
        if len(trigger_idx) > 0:  # 否则len应该为1
            # 有trial开始trigger则进行计算
            cal_flag = True
            trial_start_trig_pos = trigger_idx[0]
            # 从trial开始的位置拼接数据
            self.cache_data = eeg_data[:, trial_start_trig_pos: eeg_data.shape[1]]  # 切片含左不含右
        else:
            # 没有trial开始trigger则
            cal_flag = False
            self.__clear_cache()
        return cal_flag

    def __cal_proc(self, data_model, sum, trial_count):
        # 脑电数据+trigger
        data = data_model.data
        personID = data_model.subject_id
        # 获取trigger导
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]
        # 获取脑电数据
        eeg_data = data[0: -1, :]
        # 如果trigger为空，表示依然在当前试次中，根据数据长度判断是否计算
        if len(trigger_idx) == 0:
            # 当已缓存的数据大于等于所需要使用的计算数据时，进行计算
            if self.cache_data.shape[1] >= self.cal_len:
                # 获取所需计算长度的数据
                # self.cache_data = self.cache_data[:, self.cache_data.shape[1] - 128:]  # 去掉前面44ms不要
                # self.cache_data = self.cache_data[:, :]
                # 滤波处理
                use_data, sum, trial_count = self.__preprocess(self.cache_data, sum, trial_count)
                # 开始计算，返回计算结果
                result = recognize(use_data, personID)
                # 停止计算模式
                cal_flag = False
            else:
                # 反之继续采集数据
                self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
                result = None
                cal_flag = True
        # 下一试次已经开始,需要强制结束计算
        else:
            # 下一个trial开始trigger的位置
            next_trial_start_trig_pos = trigger_idx[0]
            # 如果拼接该数据包中部分的数据后，可以满足所需要的计算长度，则拼接数据达到所需要的计算长度
            # 如果拼接完该trial的所有数据后仍无法满足所需要的数据长度，则只能使用该trial的全部数据进行计算
            use_len = min(next_trial_start_trig_pos, self.cal_len - self.cache_data.shape[1])  # 只有小的那个能被满足
            self.cache_data = np.append(self.cache_data, eeg_data[:, 0: use_len], axis=1)  # 新读出来的数据取需要满足的那部分
            # 滤波处理
            use_data = self.__preprocess(self.cache_data)
            # 开始计算
            result = recognize(use_data, personID)
            # 开始新试次的计算模式
            cal_flag = True
            # 清除缓存的数据
            self.__clear_cache()  # 都是之前的数据用完了要清理掉
            # 添加新试次数据
            self.cache_data = eeg_data[:, next_trial_start_trig_pos: eeg_data.shape[1]]
        return cal_flag, result, sum, trial_count

    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q = 35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __clear_cache(self):
        self.cache_data = np.zeros((64, 0))

    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], 'bandpass')
        return b, a

    def Euclidean_alignment(self, R, Xi_all):
        v, Q = la.eig(R)
        V = np.diag(v ** (-0.5))
        R_1_2 = Q * V * la.inv(Q)
        Xi_all_reture = []
        for Xi in Xi_all:
            Xi_all_reture.append(np.dot(R_1_2, Xi))
        return Xi_all_reture

    def __preprocess(self, data, sum_all, trial_number):
        # 选择使用的导联
        channel_selection_index = [24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36,
                                   37, 38, 39, 42, 43, 44, 45, 46, 47]
        # channel_names = ['FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'T8', 'CP1', 'CP2', 'CP3',
        #                  'CP4', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7']
        # ch_types = list(np.full(len(channel_names), 'eeg'))
        # sample_freq = 250  # Hz
        data = np.array(data)[channel_selection_index]
        data_detrend = signal.detrend(data, axis=-1, type='linear')
        # info = mne.create_info(channel_names, sample_freq, ch_types)
        # raw = mne.io.RawArray(data, info)
        # raw.plot_psd()
        # raw.plot(duratvion=1, n_channels=len(channel_names), clipping=None, scalings={'eeg': 100})
        # raw = raw.filter(l_freq=8, h_freq=26)
        # raw = raw.filter(l_freq=4, h_freq=30)
        # raw.notch_filter(freqs=50)
        # raw.plot_psd()
        # raw.plot(duration=1, n_channels=len(channel_names), clipping=None, scalings={'eeg': 170})
        # filter_data, times = raw[:, :]
        b, a = self.butter_bandpass(8, 30, 250, 6)
        filter_data = signal.filtfilt(b, a, data_detrend, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)
        # raw = mne.io.RawArray(filter_data, info)
        # raw.plot(duration=1, n_channels=len(channel_names), clipping=None, scalings={'eeg': 170})
        used_data = []
        sum = np.zeros((len(channel_selection_index), len(channel_selection_index)))
        for i in range(6):
            Xi = np.mat(filter_data[:, (i + 2) * 6:(i + 2) * 6 + 128])
            sum = sum + np.dot(Xi, Xi.T)
            used_data.append(filter_data[:, (i + 2) * 6:(i + 2) * 6 + 128])
        sum_all = sum_all + sum
        trial_number = trial_number + len(used_data)
        R = sum_all / trial_number
        used_data = self.Euclidean_alignment(R, used_data)
        return used_data, sum_all, trial_number
