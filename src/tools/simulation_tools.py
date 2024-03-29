import numpy as np
from scipy.optimize import linear_sum_assignment

def str_to_enum(str_val, enum):
    if str_val not in enum.__members__:
        list_of_names = [name for name, member in enum.__members__.items()]
        message = '{s} is not supported.\nSupported types are:\n\t{members}'.format(s=str_val,
                                                                                    members='\n\t'.join(list_of_names))
        raise ValueError(message)
    return enum[str_val]

def find_min_err_label_permutation(l, l0, K, K0):
    e = 0
    C = np.zeros((K, K0))
    E = np.zeros((K, K0))
    for k0 in range(K0):
        for k in range(K):
            C[k, k0] = np.sum(l0[l == k] == k0)
            E[k, k0] = np.sum(l0[l == k] != k0)

    l_in, l_out = linear_sum_assignment(E)
    l_min = l_out[l]
    return l_min



def calcNErr(l, l0, K, K0):
    e = 0
    C = np.zeros((K0, K))
    E = np.zeros((K0, K))
    for k0 in range(K0):
        for k in range(K):
            C[k0, k] = np.sum(l0[l == k] == k0)
            E[k0, k] = np.sum(l0[l == k] != k0)

    rk0 = list(range(K0))
    rk = list(range(K))
    l0_to_l = []
    while len(rk) > 0:
        if len(rk0) > 0:
            C_ = C[rk0, :][:, rk]
            E_ = E[rk0, :][:, rk]
            max_corr = np.max(C_, axis=None)
            k0_corr, k_corr = np.where(C_ == max_corr)
            i_min_err = np.argmin(E_[k0_corr, k_corr])
            # mink, mink_ = np.unravel_index(np.argmax(C_, axis=None), C_.shape)
            k0_max_min = k0_corr[i_min_err]
            k_max_min = k_corr[i_min_err]
            e = e + E_[k0_max_min, k_max_min]
            l0_to_l.append((rk0[k0_max_min], rk[k_max_min]))
            rk0.pop(k0_max_min)
            rk.pop(k_max_min)
        else:
            e = e + np.sum(l == rk[0])
            rk.pop(0)
    return e, float(np.sum(np.bitwise_or(l < 0, l >= K0)))

def args_to_pid(args):
    if args.pop(0) == '-m':
        args.pop(0)

    if len(args) > 0:
        pid_offset = int(args.pop(0))
    else:
        pid_offset = 0

    if len(args) > 0:
        rel_pid = int(args.pop(0))
    else:
        rel_pid = 0

    pid = pid_offset + rel_pid
    return pid

def args_to_pid_and_sim_id(args):
    if args.pop(0) == '-m':
        args.pop(0)

    if len(args) > 0:
        pid_offset = int(args.pop(0))
    else:
        pid_offset = 0

    if len(args) > 0:
        pid_arg = args.pop(0)
        split_pid = pid_arg.split('-')
        if len(split_pid) > 1:
            rel_pid_start = int(split_pid[0])
            rel_pid_stop = int(split_pid[1])
        else:
            rel_pid_start = int(pid_arg)
            rel_pid_stop = rel_pid_start + 1
    else:
        rel_pid_start = 0
        rel_pid_stop = 1

    if len(args) > 0:
        sim_id = int(args.pop(0))
    else:
        sim_id = 1

    pid = range(pid_offset + rel_pid_start, pid_offset + rel_pid_stop)

    return pid, sim_id
