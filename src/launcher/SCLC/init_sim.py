import json
import os

import numpy as np

from src.graphs import graph_factory
from src.launcher.SCLC import constants
from src.node_classification import SpectralLearning
from src.tools.graph_tools import select_labels


def make_result_dirs():
    print('making result and plot directories')
    from pathlib import Path
    Path(constants.results_dir['init_sim']).mkdir(parents=True, exist_ok=True)
    Path(constants.plots_dir['init_sim']).mkdir(parents=True, exist_ok=True)


def combine_results():
    print('combining results')
    from src.tools.combine_results import combine_results as cr
    cr(constants.results_dir['init_sim'],has_lists=True)


def plot():
    print('plotting')
    import pandas as pd
    import matplotlib.pyplot as plt
    result_dir = constants.results_dir['init_sim']
    results_file_name = os.path.join(result_dir,'comb.json')
    print('loading {f}'.format(f=results_file_name))
    results_df = pd.read_json(results_file_name)
    print('done loading')
    decimals = pd.Series([2], index=['eps'])
    results_df.round(decimals)
    grouped_results = results_df.set_index(['num_classes','percentage_labeled','eps','i_rep']).sort_index()

    list_df = pd.DataFrame(grouped_results.objective.tolist(),index=grouped_results.index)

    for (num_classes, eps, percentage_labeled), pl_df in list_df.groupby(level=['num_classes', 'eps', 'percentage_labeled']):
        pl_df = pl_df.droplevel(['num_classes', 'eps', 'percentage_labeled'])
        fig_name = 'conv_K{k}_eps{eps}_pl{pl}'.format(k=num_classes,eps=eps,pl=percentage_labeled)
        print('plotting {s}'.format(s=fig_name))
        min_ = np.min(np.array(pl_df), axis=0)
        max_ = np.max(np.array(pl_df), axis=0)
        diff = max_[-1]-min_[-1]
        pl_df.reindex(index=pl_df.index[::-1])
        limits = [[max_[-1]-0.02, min_[-1]-3*diff],[max_[-1]+0.005,max_[-1]+diff/2]]
        if num_classes == 3:
            limits = [[3.64],[3.79]]
        elif num_classes == 5:
            limits = [[5.63],[5.73]]
        elif num_classes == 10:
            limits = [[10.23],[10.33]]
        for i, (y_min, y_max) in enumerate(zip(*limits)):
            for i_rep in reversed(pl_df.index):
                rep = pl_df.loc[i_rep]
                if i_rep == 0:
                    plt.semilogx(rep,'k')
                else:
                    plt.semilogx(rep,'darkgrey')
            x_max = 20000
            plt.ylim([y_min, y_max])
            plt.xlim([10, x_max])
            config={'ylim':[y_min, y_max],
                    'xlim':[10, x_max],
                    'max': max_[-1],
                    'min': min_[-1]}
            axis_filename = os.path.join(constants.plots_dir['init_sim'],'axis_{n}_scale_{i}.png'.format(i=i,n=fig_name))
            filename = os.path.join(constants.plots_dir['init_sim'],'{n}_scale_{i}.png'.format(i=i,n=fig_name))
            config_filename = os.path.join(constants.plots_dir['init_sim'],'config_{n}_scale_{i}.txt'.format(i=i,n=fig_name))
            plt.savefig(axis_filename)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(filename)
            plt.close()
            with open(config_filename,'w') as config_file:
                json.dump(config,config_file)


def print_num_configs():
    _, _, num_graph_config = get_config(0)
    print('{n} configs'.format(n=num_graph_config))


def print_all_configs():
    _, _, num_graph_config = get_config(0)
    for pid in range(num_graph_config):
        print('pid={p}'.format(p=pid))
        _, _, _ = get_config(pid)


def get_config(pid, sim_id, suppress_output=False):
    num_classes_list = [3, 5, 10]
    t_max_list = [20000, 20000, 20000]
    eps_list = [0.4]  # np.linspace(0, 0.5, 11)
    percentage_labeled_list = [1, 5, 10]
    assert len(num_classes_list) == len(t_max_list)

    len_num_classes = len(num_classes_list)
    len_eps = len(eps_list)
    len_percentage = len(percentage_labeled_list)
    num_configs = len_num_classes * len_eps * len_percentage

    assert sim_id < num_configs

    num_classes = num_classes_list[sim_id % len_num_classes]
    t_max = t_max_list[sim_id % len_num_classes]
    sim_id = sim_id // len_num_classes
    eps = eps_list[sim_id % len_eps]
    sim_id = sim_id // len_eps
    percentage_labeled = percentage_labeled_list[sim_id % len_percentage]
    sim_id = sim_id // len_percentage

    num_nodes = 300*num_classes
    pi = min(1, num_classes * (num_classes + 2) / num_nodes)
    graph_config = {'model': 'SBM',
                    'type': 'Bernoulli',
                    'class_distribution': [1] * num_classes,
                    'num_nodes': num_nodes,
                    'pi': pi,
                    'pe': pi,
                    'li': 0.5 - eps,
                    'le': 0.5 + eps}
    use_det = pid == 0
    sim_config = {'use_det': use_det,
                  'i_rand_rep': pid,
                  'eps': eps,
                  't_max': t_max,
                  'percentage_labeled': percentage_labeled}
    if not suppress_output:
        print('i_config: {i}'.format(i=sim_id))
        print('classifying SBM with {n} classes and eps={eps} making use of {p:.0%} labels'.format(n=num_classes, eps=eps, p=percentage_labeled / 100))
        print('deterministic initialization = {d}'.format(d='True' if use_det else 'False'))
    return graph_config, sim_config, num_configs


def run(pid, sim_id=1):
    print('pid: {p}'.format(p=pid))
    print('sim_id: {s}'.format(s=sim_id))

    results = {'objective': [],
              'i_rep': [],
              'num_classes': [],
              'eps': [],
              'percentage_labeled': []}

    graph_config, sim_config, num_graph_config = get_config(pid, sim_id)
    eps = sim_config['eps']
    percentage_labeled = sim_config['percentage_labeled']
    use_det = sim_config['use_det']
    i_rand_rep = sim_config['i_rand_rep']
    t_max = sim_config['t_max']

    graph_seed = 1
    print('graph_seed={s}'.format(s=graph_seed))
    np.random.seed(graph_seed)
    graph = graph_factory.make_graph(**graph_config)
    labels = select_labels(graph, label_amount=percentage_labeled, is_percentage=True, sorting_level=1)

    np.random.seed(i_rand_rep)
    sclc = SpectralLearning(num_classes=graph.num_classes, objective='BNC_INDEF', multiclass_method='joint', random_init=not use_det, save_intermediate=True, eps=0, t_max=t_max, verbosity=0)
    l_est = sclc.estimate_labels(graph, labels=labels)
    x = sclc.embedding
    (objective_values, objective_matrix) = sclc.intermediate_results[1:]
    n_err = np.sum(l_est != graph.class_labels)
    x_round = np.zeros((graph.num_nodes, graph.num_classes))
    x_round[range(graph.num_nodes),l_est] = 1
    x_round = x_round/np.linalg.norm(x_round,axis=0,keepdims=True)
    obj_round = np.trace(x_round.T.dot(objective_matrix.dot(x_round)))
    print('n_err={n}, obj={o}, rounded_obj={o_r}'.format(n=n_err, o=objective_values[-1], o_r=obj_round))
    results['objective'].append(objective_values)
    results['i_rep'].append(i_rand_rep)
    results['num_classes'].append(graph.num_classes)
    results['eps'].append(eps)
    results['percentage_labeled'].append(percentage_labeled)

    results_dir = constants.results_dir['init_sim']
    results_file_name = os.path.join(results_dir, '{sid}_{pid}.json'.format(sid=sim_id,pid=pid))
    with open(results_file_name, 'w') as results_file:
        json.dump(results, results_file)


if __name__ == '__main__':
    import sys
    from src.tools.simulation_tools import args_to_pid_and_sim_id

    args = sys.argv

    if args[1].startswith('-'):
        if args[1] == '-mk':
            make_result_dirs()
        elif args[1] == '-c':
            combine_results()
        elif args[1] == '-p':
            plot()
        else:
            print('unknown command {c}'.format(c=args[1]))
        pass
    else:
        pid, sim_id = args_to_pid_and_sim_id(sys.argv)
        run(pid, sim_id)
