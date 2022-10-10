import json
import os

import numpy as np

from src.launcher import SBMSimulation
from src.launcher.LSBM_MAP import constants
from src.node_classification import SpectralLearning, TvConvex


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

    plots_folder = constants.plots_dir['sbm_sim']
    for root, subdirs, files in os.walk(plots_folder):
        for x_json_name in files:
            if x_json_name.startswith("x") and x_json_name.endswith(".json"):
                x_csv_name = x_json_name[:-4]+'csv'
                with open(os.path.join(plots_folder,x_json_name)) as x_file:
                    x = json.load(x_file)

                x_array_style = {}
                for key, val in x.items():
                    x_array = np.array(val)
                    x_array = np.sort(x_array,axis=0)
                    for n in range(x_array.shape[1]):
                        x_array_style['{k}{n}'.format(k=key.replace('_',''), n=n)] = x_array[:, n]
                x_df = pd.DataFrame(x_array_style)
                x_df.to_csv(os.path.join(plots_folder,x_csv_name))


def get_graph_config_lists(sim_id):
    scale_pi = None
    scale_pe = None
    eps_list = np.linspace(0, 0.5, 11)
    percentage_labeled_list = [10]
    if sim_id == 0:
        num_classes_list = [2, 3, 5, 10]
        num_nodes_list = [3000]*4
    else:
        raise ValueError('unknown sim_id')
    class_distribution_list = [[1] * nc for nc in num_classes_list]

    sbm_config_dict = {'num_classes': num_classes_list,
                       'num_nodes': num_nodes_list,
                       'class_distribution': class_distribution_list}
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
            {'name': 'snc', 'method': SpectralLearning(num_classes=num_classes, objective='BNC_INDEF')},
            {'name': 'tv', 'method': TvConvex(num_classes=num_classes, verbosity=1)},
            {'name': 'tv_reg', 'method': TvConvex(num_classes=num_classes, verbosity=1, do_regularize=True)},
        ]
    else:
        raise ValueError('unknown sim_id')

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
            for i in range(1):
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
