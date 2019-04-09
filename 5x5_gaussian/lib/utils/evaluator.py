import numpy as np

def check_mode(pc, data, th):
    '''
    input- (250, 2)
    output:
    - valid_mode: (n*n)
    '''
    S = np.shape(data)[0]
    valid_mode = np.zeros(S)
    for index in range(S):
        x = data[index][0]
        y = data[index][1]
        valid_x = np.logical_and((pc[:,0] > x - th), (pc[:,0] < x + th))
        valid_y = np.logical_and((pc[:,1] > y - th), (pc[:,1] < y + th))
        valid = np.logical_and(valid_x, valid_y)
        valid_mode[int(index)] = np.any(valid)
    return valid_mode

def check_quality(pc, data, th):
    '''
    input- (250, 2)
    output:
    - count_quality: (1)
    '''
    S = np.shape(data)[0]
    N = np.shape(pc)[0]
    res = np.zeros(N)
    for index in range(S):
        x = data[index][0]
        y = data[index][1]
        valid_x = np.logical_and((pc[:,0] > x - th), (pc[:,0] < x + th))
        valid_y = np.logical_and((pc[:,1] > y - th), (pc[:,1] < y + th))
        valid = np.logical_and(valid_x, valid_y)
        res = np.logical_or(res, valid)
    count_quality = res.sum()
    return count_quality


