import json
import os

import numpy as np

from src.launcher import SBMSimulation
from src.launcher.LSBM_MAP import constants
from src.node_classification import LsbmMap, LsbmMlLelarge, SbmMlHajek, SbmSpYun, Sponge


def make_result_dirs():
    print('making result and plot directories')
    from pathlib import Path
    for dir in constants.results_dir['sbm_sim']:
        Path(dir).mkdir(parents=True, exist_ok=True)
    Path(constants.plots_dir['sbm_sim']).mkdir(parents=True, exist_ok=True)


def combine_results():
    print('combining results')
    from src.tools.combine_results import combine_results as cr
    for dir in constants.results_dir['sbm_sim']:
        cr(dir, has_lists=False)


def plot():
    print('plotting')
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import itertools

    groups = [['eps', 'scale_pi'],
              ['eps', 'num_classes'],
              ['eps', 'num_classes']]

    for sim_id in range(3):
        results_file_name = os.path.join(constants.results_dir['sbm_sim'][sim_id], 'comb.json')
        with open(results_file_name) as results_file:
            results = json.load(results_file)

        del results['graph_config']
        results_df = pd.DataFrame(results)

        mean_results = results_df.groupby(groups[sim_id]).mean().reset_index(level=list(range(1, len(groups[sim_id]))))
        unique_lists = []
        for grouping in groups[sim_id][1:]:
            unique_lists.append(results_df[grouping].unique())
        for ind in itertools.product(*unique_lists):
            mask = np.ones(mean_results.shape[0], dtype=bool)
            csv_file_name = 'sbm_mean_sim_{sid}'.format(sid=sim_id)
            for key, val in zip(groups[sim_id][1:], ind):
                mask = np.bitwise_and(mask, mean_results[key] == val)
                csv_file_name += '_{k}_{v}'.format(k=key, v=val)
            csv_file_name += '.csv'
            subdf = mean_results[mask]
            subdf.to_csv(os.path.join(constants.plots_dir['sbm_sim'], csv_file_name))

        names = [col[len('n_err_unlabeled') + 1:] for col in results_df.columns if col.startswith('n_err_unlabeled')]
        global_cols = groups[sim_id]
        name_dfs = []
        for name in names:
            name_cols = [col for col in results_df.columns if col.endswith(name)]
            cols = [col[:-len(name) - 1] for col in name_cols]
            name_df = results_df[global_cols + name_cols]
            name_df.columns = global_cols + cols

            pd.options.mode.chained_assignment = None
            name_df.loc[:, 'name'] = name
            pd.options.mode.chained_assignment = 'warn'

            name_dfs.append(name_df)
            pass
        results_df = pd.concat(name_dfs, ignore_index=True)
        results_mean = results_df.groupby(['name'] + groups[sim_id]).mean().reset_index()

        sns.lineplot(data=results_mean, x='eps', y='n_err_unlabeled', hue=groups[sim_id][1], style='name')
        plt.show()
        sns.lineplot(data=results_mean, x='eps', y='t_run', hue=groups[sim_id][1], style='name')
        plt.show()

    # x_filename = os.path.join(constants.plots_dir['sbm_sim'], 'x.json')
    # with open(x_filename) as x_file:
    #     x = json.load(x_file)
    #
    # x_array_style = {}
    # for key, val in x.items():
    #     x_array = np.array(val)
    #     for n in range(x_array.shape[1]):
    #         x_array_style['{k}{n}'.format(k=key, n=n)] = x_array[:, n]
    # x_df = pd.DataFrame(x_array_style)
    # x_df.to_csv(os.path.join(constants.plots_dir['sbm_sim'], 'x.csv'))


def get_graph_config_lists(sim_id):
    scale_pi = None
    scale_pe = None
    if sim_id == 0:
        num_classes_list = [2, 2, 2]
        num_nodes_list = [1000, 1000, 1000]
        class_distribution_list = [[1, 1], [1, 1], [1, 1]]
        percentage_labeled_list = [0]
        scale_pi = [3 / 8, 6 / 8, 12 / 8]
        scale_pe = [1 / 8, 4 / 8, 8 / 8]
        eps_list = np.linspace(0, 0.5, 11)
    elif sim_id == 1:
        num_classes_list = [2, 5, 8]
        num_nodes_list = [120]*3
        class_distribution_list = [[1] * nc for nc in num_classes_list]
        eps_list = np.linspace(0, 0.5, 11)
        percentage_labeled_list = [0]
    else:
        num_classes_list = [2, 5, 10]
        num_nodes_list = [1000]*3
        class_distribution_list = [[1] * nc for nc in num_classes_list]
        eps_list = np.linspace(0, 0.5, 11)
        percentage_labeled_list = [0]

    if scale_pi is None:
        scale_pi = [1] * len(num_classes_list)
    if scale_pe is None:
        scale_pe = [1] * len(num_classes_list)

    sbm_config_dict = {'num_classes': num_classes_list,
                       'num_nodes': num_nodes_list,
                       'class_distribution': class_distribution_list,
                       'scale_pi': scale_pi,
                       'scale_pe': scale_pe}
    # for t in zip(*sbm_config_dict.values()) produces one slice over all lists
    # dict(zip(sbm_config_dict.keys(),t)) combines each element in the slice with the corresponding key
    # dict(zip(sbm_config_dict,t)) produces the same result -- we include .keys() for more clarity
    sbm_config_list = [dict(zip(sbm_config_dict.keys(), t)) for t in zip(*sbm_config_dict.values())]

    config_lists = {'eps_list': eps_list,
                    'percentage_labeled_list': percentage_labeled_list,
                    'sbm_config_list': sbm_config_list}
    return config_lists


def get_methods(graph_config, sim_id):
    pi = graph_config['pi']
    pe = graph_config['pe']
    li = graph_config['li']
    le = graph_config['le']
    num_classes = len(graph_config['class_distribution'])
    class_distribution = graph_config['class_distribution']
    eps = 1e-5

    if sim_id == 0:
        methods = [
            {'name': 'lsbm_ml_lelarge', 'is_unsupervised': True,
             'method': LsbmMlLelarge(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes, verbosity=1)},
            {'name': 'lsbm_map', 'is_unsupervised': True, 'l_guess': 'lsbm_ml_lelarge',
             'method': LsbmMap(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes, verbosity=1,
                               class_distribution=class_distribution, eps=eps, t_max=1e5)},
        ]
    elif sim_id == 1:
        methods = [
            {'name': 'sponge', 'is_unsupervised': True,
             'method': Sponge(num_classes=num_classes)},
            {'name': 'sbm_ml_hajek', 'is_unsupervised': True,
             'method': SbmMlHajek(num_classes=num_classes, class_distribution=class_distribution)},
            {'name': 'lsbm_map', 'is_unsupervised': True, 'l_guess': 'sponge',
             'method': LsbmMap(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes, verbosity=1,
                               class_distribution=class_distribution, eps=eps, t_max=1e5)},
        ]
    elif sim_id == 2:
        methods = [
            {'name': 'sponge', 'is_unsupervised': True,
             'method': Sponge(num_classes=num_classes)},
            {'name': 'sbm_sp_yun', 'is_unsupervised': True,
             'method': SbmSpYun(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes, class_distribution=class_distribution)},
            {'name': 'sbm_sp_yun_sponge', 'is_unsupervised': True, 'l_guess': 'sponge',
             'method': SbmSpYun(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes, class_distribution=class_distribution)},
            {'name': 'lsbm_map', 'is_unsupervised': True, 'l_guess': 'sponge',
             'method': LsbmMap(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes, verbosity=1,
                               class_distribution=class_distribution, eps=eps, t_max=1e5)},
        ]
    else:
        methods = [
            {'name': 'lsbm_map', 'method': LsbmMap(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes,
                                                   class_distribution=class_distribution, eps=eps, t_max=1e5,
                                                   verbosity=1)},
        ]
    return methods


def run(pid, sim_id):
    config_lists = get_graph_config_lists(sim_id)

    sim = SBMSimulation(**config_lists)
    graph_config, percentage_labeled, is_percentage = sim.get_graph_config(pid)
    print('sbm with pi={pi} (a={a}) and pe={pe} (b={b})'.format(pi=graph_config['pi'],
                                                                a=graph_config['num_nodes'] * graph_config['pi'],
                                                                pe=graph_config['pe'],
                                                                b=graph_config['num_nodes'] * graph_config['pe']))
    method_configs = get_methods(graph_config, sim_id)
    sim.add_method(method_configs)
    sim.run_simulation(pid)
    sim.save_results(constants.results_dir['sbm_sim'][sim_id], split_file=False)

    if pid < len(sim.graph_config_list):
        filename = os.path.join(constants.plots_dir['sbm_sim'],
                                'x_s{sid}_p{pid}.json'.format(sid=sim_id, pid=pid))
        x_lists = {k: v.tolist() for k, v in sim.embedding.items()}
        with open(filename, 'w') as x_file:
            json.dump(x_lists, x_file)


if __name__ == '__main__':
    import sys
    from src.tools.simulation_tools import args_to_pid_and_sim_id

    args = sys.argv

    if args[1].startswith('-'):
        if args[1] == '-mk':
            make_result_dirs()
        elif args[1] == '-c':
            combine_results()
        elif args[1] == '-n':
            for i in range(3):
                config_lists = get_graph_config_lists(sim_id=i)
                sim = SBMSimulation(**config_lists)
                print('{n} configs in simulation {i}'.format(n=len(sim.graph_config_list), i=i))
        elif args[1] == '-p':
            plot()
        else:
            print('unknown command {c}'.format(c=args[1]))
        pass
    else:
        pid, sim_id = args_to_pid_and_sim_id(sys.argv)
        run(pid, sim_id)
