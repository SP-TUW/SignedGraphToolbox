import numpy as np

def str_to_enum(str_val, enum):
    if str_val not in enum.__members__:
        list_of_names = [name for name, member in enum.__members__.items()]
        message = '{s} is not supported.\nSupported types are:\n\t{members}'.format(s=str_val,
                                                                                    members='\n\t'.join(list_of_names))
        raise ValueError(message)
    return enum[str_val]

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

def args_to_pid(args, get_sim_name=False):
    if args.pop(0) == '-m':
        args.pop(0)

    if get_sim_name:
        if len(args) > 0:
            sim_name = args.pop(0)
        else:
            sim_name = 'n_err'

    if len(args) > 0:
        pid_offset = int(args.pop(0))
    else:
        pid_offset = 0

    if len(args) > 0:
        rel_pid = int(args.pop(0))
    else:
        rel_pid = 0

    pid = pid_offset + rel_pid
    if get_sim_name:
        return pid, sim_name
    else:
        return pid

def args_to_pid_and_rep(args):
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

    if len(args) > 0:
        sim_rep = int(args.pop(0))
    else:
        sim_rep = 1

    pid = pid_offset + rel_pid

    return pid, sim_rep
