import numpy as np
import scipy.io as scio
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import time
from numpy.linalg import inv
from train import *
from result_analysis import get_table

def load_data(data, N):
    '''

    :param data: input data
    :param N: return N lowest and highest data points
    :return:
    '''
    tmp = int(N/2)
    # print(tmp)
    data = np.concatenate((data[:tmp], data[-tmp:]), axis=0)
    # print(data.shape)

    data[:tmp, -1] = 1
    data[tmp:, -1] = 0

    # print(data.shape)
    data_x = data[:, :-1]
    data_y = data[:, -1]

    return data_x, data_y

def cal_prop(odfs, p):
    props = []
    for i in range(odfs.shape[0]):
        c = p.dot(odfs[i]).reshape(6, 6)
        # print(c.shape)
        s = inv(c)
        e11 = -(1/s[0,0])
        # print(e11)
        if e11 < 0:
            e11 = -e11
        props.append(e11)
    return np.array(props).reshape(-1, 1)

def random_data_sampling(q, iter_num):
    odfs = []
    for num in range(2, 40, 2):
        # print(q.shape)
        for i in range(iter_num):
            odf = np.zeros(q.shape[0])
            randSep = np.random.random(num-1)
            randSep = np.sort(randSep)
            randSep = np.insert(randSep, 0, 0)
            randSep = np.append(randSep, 1)
            # print(randSep)
            randIntvl = np.diff(randSep, n=1, axis=0)
            randInd = np.random.permutation(range(num))
            odf[randInd] = randIntvl / q[randInd, 0]
            # print(odf.shape)
            s = np.dot(odf, q)
            if s == 1:
                odfs.append(odf)
    return odfs

def generate_odf(q, randInd, randIntvl):
    odf = np.zeros(q.shape[0])
    odf[randInd] = randIntvl / q[randInd, 0]
    # print(odf.shape)
    s = np.dot(odf, q)
    return odf, s


def first_data_sampling(q, iter_num, sorted_ind):
    odfs = []
    for k in range(10, 20, 5):
        for num in range(2, 11):
            # print(q.shape)
            for i in range(iter_num):
                randSep = np.random.random(num-1)
                randSep = np.sort(randSep)
                randSep = np.insert(randSep, 0, 0)
                randSep = np.append(randSep, 1)
                # print(randSep)
                randIntvl = np.diff(randSep, n=1, axis=0)
                randInd = np.random.choice(sorted_ind[:k], num)
                odf, s = generate_odf(q, randInd, randIntvl)
                if s == 1:
                    odfs.append(odf)

                randInd = np.random.choice(sorted_ind[-k:], num)
                odf, s = generate_odf(q, randInd, randIntvl)
                if s == 1:
                    odfs.append(odf)

    return odfs

def second_data_sampling(q, iter_num, sorted_ind):
    odfs = []
    for k in range(10, 30, 5):
        for num in range(2, 11):
            # print(q.shape)
            for i in range(iter_num):
                odf = np.zeros(q.shape[0])
                randSep = np.random.random(num-1)
                randSep = np.sort(randSep)
                randSep = np.insert(randSep, 0, 0)
                randSep = np.append(randSep, 1)
                # print(randSep)
                randIntvl = np.diff(randSep, n=1, axis=0)
                # print(randIntvl.shape)
                # randInd = randperm(50)
                # randInd = np.random.permutation(range(50))
                randInd = np.random.choice(sorted_ind[:k], num)
                odf[randInd] = randIntvl / q[randInd, 0]
                # print(odf.shape)
                s = np.dot(odf, q)
                if s == 1:
                    odfs.append(odf)

    return odfs

def single_crystal(odfs, p):
    property = cal_prop(odfs, p)

    in_max = property.argmax(axis=0)[-1]
    print('single max ind: {}, property: {}'.format(in_max, property[in_max][0]))

    in_min = property.argmin(axis=0)[-1]
    print('single min ind: {}, property: {}'.format(in_min, property[in_min][0]))

    sorted_ind = np.argsort(property[:, 0])
    # print(sorted_ind)

    return sorted_ind

if __name__ == '__main__':

    # first_sampling = np.load('data_sampling/rand_data_sampling.npy')
    # print(first_sampling.shape)

    id_list = [1028, 1029, 1030, 14732, 14815, 84837, 84936]
    threshold = [0.0001, 0.001, 0.005, 0.01]

    N = 500000
    zero_iter_num = 1000000
    first_iter_num = 100000
    second_iter_num = int(1000000/2)

    save_ret_num = 100000

    # # test parameters
    # N = 1000
    # zero_iter_num = 10000
    # first_iter_num = 1000
    # second_iter_num = 10000

    property_name = 'properties' + str(1028) + '.mat'
    q = scio.loadmat(property_name)['volumefraction'].T

    # # zero data sampling
    # zero_time = time.time()
    # odfs = random_data_sampling(q, iter_num=zero_iter_num)
    # np.save('zero_data_sampling_v2.npy', odfs)
    # print('zero data sampling time: {}'.format(time.time()-zero_time))
    #
    # # save initial first data sampling
    # odfs = np.load('zero_data_sampling_v2.npy')
    # np.save('first_data_sampling_v2.npy', odfs)

    # print single crystal results
    single_odfs = np.zeros((q.shape[0], q.shape[0]))
    for i in range(q.shape[0]):
        single_odfs[i, i] = 1/q[i]

    for id in id_list:
        start = time.time()

        id = str(id)
        print('id: ', id)
        property_name = 'properties' + id + '.mat'
        q = scio.loadmat(property_name)['volumefraction'].T
        p = scio.loadmat(property_name)['stiffness']

        # calculate single_crystal
        single_sorted_ind = single_crystal(single_odfs, p)
        # print(single_sorted_ind)

        # first data sampling
        first_time = time.time()
        odfs = np.load('first_data_sampling_v2.npy')

        # tmp = first_data_sampling(q, iter_num=first_iter_num, sorted_ind=single_sorted_ind)
        # odfs = np.concatenate((odfs, np.array(tmp)), axis=0)
        # np.save('first_data_sampling_v2.npy', odfs)
        # print('first data sampling time: {}'.format(time.time() - first_time))

        # ML model
        ML_time = time.time()
        props = cal_prop(odfs, p)
        data = np.concatenate((odfs, props), axis=1)
        data = data[data[:, -1].argsort()]
        data_x, data_y = load_data(data, N)

        feature_scores, sorted_feature_ids = feature_selection(data_x, data_y)
        feature_ranges = calc_feature_ranges(data_x, data_y)
        print(' ML method cost %.4f seconds' % (time.time() - ML_time))

        # print(sorted_feature_ids)

        # second sampling
        second_time = time.time()

        # min
        # sorted_feature_ids - single_sorted_ind[:10]
        min_sorted_feature_ids = [i for i in list(sorted_feature_ids) if i not in list(single_sorted_ind[-10:])]
        min_sorted_feature_ids = np.array(min_sorted_feature_ids)
        # print(min_sorted_feature_ids.shape)
        # print(min_sorted_feature_ids)
        min_second_samping_odfs = second_data_sampling(q, iter_num=second_iter_num, sorted_ind=min_sorted_feature_ids)
        min_second_samping_odfs = np.array(min_second_samping_odfs)
        # print(second_samping_odfs.shape)

        # max
        # sorted_feature_ids - single_sorted_ind[-10:]
        max_sorted_feature_ids = [i for i in list(sorted_feature_ids) if i not in list(single_sorted_ind[:10])]
        max_sorted_feature_ids = np.array(max_sorted_feature_ids)
        # print(min_sorted_feature_ids.shape)
        # print(min_sorted_feature_ids)
        max_second_samping_odfs = second_data_sampling(q, iter_num=second_iter_num, sorted_ind=max_sorted_feature_ids)
        max_second_samping_odfs = np.array(max_second_samping_odfs)

        second_samping_odfs = np.concatenate((min_second_samping_odfs, max_second_samping_odfs), axis=0)
        print(second_samping_odfs.shape)
        print('second data sampling time: {}'.format(time.time() - second_time))

        # data analysis
        opt_file = id + 'optimization_results.mat'
        optimization_r = scio.loadmat(opt_file)
        min_E11, max_E11 = optimization_r['min_E11'], optimization_r['max_E11']
        print('fmincon: ', min_E11, max_E11)

        # get tables
        # target_ret = np.load('target_result2/' + id + 'target_sampling.npy')
        # print(target_ret.shape)
        props = cal_prop(second_samping_odfs, p)
        second_data = np.concatenate((second_samping_odfs, props), axis=1)
        # target_ret = np.concatenate((second_data, target_ret), axis=0)
        target_ret = second_data
        target_ret = target_ret[np.argsort(target_ret[:, -1])]
        target_ret = np.concatenate((target_ret[:save_ret_num], target_ret[-save_ret_num:]), axis=0)
        print(target_ret.shape, target_ret[0, -1], target_ret[-1, -1])
        np.save('target_result2/' + id +'target_sampling.npy', second_data)

        ret = get_table(threshold, second_data, 'min', min_E11)
        ret = get_table(threshold, second_data, ' max', max_E11)

        print('all time: {}'.format(time.time()-start))


