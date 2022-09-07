from src.launcher.SCLC import constants
from src.launcher import SBMSimulation
from src.node_classification import SpectralLearning
import numpy as np
import json
import os


def make_result_dirs():
    print('making result and plot directories')
    from pathlib import Path
    Path(constants.results_dir['sbm_sim']).mkdir(parents=True, exist_ok=True)
    Path(constants.plots_dir['sbm_sim']).mkdir(parents=True, exist_ok=True)


def combine_results():
    print('combining results')
    from src.tools.combine_results import combine_results as cr
    cr(constants.results_dir['sbm_sim'],has_lists=False)


def plot():
    print('plotting')
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import itertools

    results_file_name = os.path.join(constants.results_dir['sbm_sim'], 'comb.json')
    with open(results_file_name) as results_file:
        results = json.load(results_file)

    del results['graph_config']
    results_df = pd.DataFrame(results)

    mean_results = results_df.groupby(['eps', 'percentage_labeled', 'num_classes']).mean().reset_index(level=[1, 2])
    num_classes_list = results_df['num_classes'].unique()
    for pl, nc in itertools.product(results_df['percentage_labeled'].unique(), results_df['num_classes'].unique()):
        subdf = mean_results[mean_results['percentage_labeled'] == pl]
        for nc in num_classes_list:
            subdf[subdf['num_classes'] == nc].to_csv(
                os.path.join(constants.plots_dir['sbm_sim'], 'sbm_mean_{n}_{pl}.csv'.format(n=nc, pl=pl)))

    names = [col[len('n_err_unlabeled') + 1:] for col in results_df.columns if col.startswith('n_err_unlabeled')]
    global_cols = ['eps', 'percentage_labeled']
    name_dfs = []
    for name in names:
        name_cols = [col for col in results_df.columns if col.endswith(name)]
        cols = [col[:-len(name) - 1] for col in name_cols]
        name_df = results_df[global_cols + name_cols]
        name_df.columns = global_cols + cols

        pd.options.mode.chained_assignment = None
        name_df.loc[:,'name'] = name
        pd.options.mode.chained_assignment = 'warn'

        name_dfs.append(name_df)
        pass
    results_df = pd.concat(name_dfs, ignore_index=True)
    results_mean = results_df.groupby(['name', 'eps', 'percentage_labeled']).mean().reset_index()

    sns.lineplot(data=results_mean, x='eps', y='n_err_unlabeled', hue='percentage_labeled', style='name')
    plt.show()
    sns.lineplot(data=results_mean, x='eps', y='t_run', hue='percentage_labeled', style='name')
    plt.show()

    x_filename = os.path.join(constants.plots_dir['sbm_sim'], 'x.json')
    with open(x_filename) as x_file:
        x = json.load(x_file)

    x_array_style = {}
    for key, val in x.items():
        x_array = np.array(val)
        for n in range(x_array.shape[1]):
            x_array_style['{k}{n}'.format(k=key, n=n)] = x_array[:, n]
    x_df=pd.DataFrame(x_array_style)
    x_df.to_csv(os.path.join(constants.plots_dir['sbm_sim'], 'x.csv'))


def get_graph_config_lists():
    num_classes_list = [3, 5, 10]
    num_nodes_list = [300*nc for nc in num_classes_list]
    class_distribution_list = [[1]*nc for nc in num_classes_list]
    eps_list = np.linspace(0, 0.5, 11)
    percentage_labeled_list = [1, 5, 10, 15]
    config_lists = {'eps_list': eps_list,
                    'percentage_labeled_list': percentage_labeled_list,
                    'sbm_config_list': [{'num_classes': cnd[0], 'num_nodes': cnd[1], 'class_distribution': cnd[2]}
                                          for cnd in zip(num_classes_list,num_nodes_list,class_distribution_list)]}
    return config_lists


def get_methods():
    methods = [{'name': 'joint', 'method': SpectralLearning(num_classes=None,multiclass_method='joint',eps=1e-10)},
               {'name': 'qr', 'method': SpectralLearning(num_classes=None, multiclass_method='qr',eps=1e-10)},
               {'name': 'seq', 'method': SpectralLearning(num_classes=None,multiclass_method='sequential',eps=1e-10)},
               {'name': 'joint_rand', 'method': SpectralLearning(num_classes=None,multiclass_method='joint',random_init=True,eps=1e-10)}]
    return methods


def run(pid):
    config_lists = get_graph_config_lists()
    method_configs = get_methods()

    sim = SBMSimulation(**config_lists)
    sim.add_method(method_configs)
    sim.run_simulation(pid)
    sim.save_results(constants.results_dir['sbm_sim'], split_file=False)

    if pid == 30:
        filename = os.path.join(constants.plots_dir['sbm_sim'],'x.json')
        x_lists = {'joint': sim.embedding['joint'].tolist(),
                   'seq': sim.embedding['seq'].tolist()}
        with open(filename,'w') as x_file:
            json.dump(x_lists,x_file)


if __name__ == '__main__':
    import sys
    from src.tools.simulation_tools import args_to_pid

    args = sys.argv

    if args[1].startswith('-'):
        if args[1] == '-mk':
            make_result_dirs()
        elif args[1] == '-c':
            combine_results()
        elif args[1] == '-n':
            config_lists = get_graph_config_lists()
            sim = SBMSimulation(**config_lists)
            print('{n} configs in this simulation'.format(n=len(sim.graph_config_list)))
        elif args[1] == '-p':
            plot()
        else:
            print('unknown command {c}'.format(c=args[1]))
        pass
    else:
        pid = args_to_pid(sys.argv)
        run(pid)
