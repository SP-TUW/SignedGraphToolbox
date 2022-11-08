import json
import os

import numpy as np

from src.launcher import SBMSimulation
from src.launcher.TV_CONVEX import constants
from src.node_classification import SpectralLearning, Sponge, TvConvex, TvNonConvex, LsbmMap


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
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import itertools
    import re
    from pathlib import Path
    import warnings
    import seaborn as sns




    plots_folder = constants.plots_dir['sbm_sim']
    # for root, subdirs, files in os.walk(plots_folder):
    #     filename_list = ['x_s{s}_p24.json'.format(s=s) for s in range(len(constants.results_dir['sbm_sim']))] + ['x_s{s}_p34.json'.format(s=s) for s in range(4)]
    #     for x_json_name in sorted(files):
    #     # for x_json_name in ['x_s0_p24.json','x_s0_p34.json','x_s0_p38.json','x_s0_p39.json']:
    #     # for x_json_name in filename_list:
    #         if x_json_name.startswith("x") and x_json_name.endswith(".json") and x_json_name in filename_list:
    #             print(x_json_name)
    #             x_name = x_json_name[:-5]
    #             x_csv_name = x_name + '.csv'
    #             with open(os.path.join(plots_folder,x_json_name)) as x_file:
    #                 x = json.load(x_file)
    #
    #             x_array_style = {}
    #             for key, val in x.items():
    #                 if key not in ['tv{e:0>2d}'.format(e=e) for e in range(0,45,5)]:
    #                     continue
    #                 # x_array = np.array(val)
    #                 # x_array = np.sort(x_array,axis=0)
    #                 # for n in range(x_array.shape[1]):
    #                 #     x_array_style['{k}{n}'.format(k=key.replace('_',''), n=n)] = x_array[:, n
    #                 x_df = pd.DataFrame(val)
    #                 x_df['max'] = x_df.idxmax(axis=1)
    #                 x_df.index = x_df.index.rename('i')
    #                 # x_df.to_csv(os.path.join(plots_folder,x_name+'_'+key+'.csv'))
    #                 num_nodes = x_df.shape[0]
    #                 s = (4-np.log10(num_nodes/9))**2
    #                 for k in range(len(val[0])):
    #                     plt.figure(figsize=(3.1/2.54, 2.0/2.54), dpi=600)
    #                     plt.scatter(x_df.index, x_df[k], c=x_df['max']==k, marker='.', s=s, cmap='PiYG', norm=colors.Normalize(vmin=-.2, vmax=1.2), linewidths=0)
    #                     # plt.show()
    #                     plt.xlim(np.array([-0.05, 1.05])*x_df.shape[0])
    #                     plt.ylim([-1.1, 1.1])
    #                     plt.axis('off')
    #                     plt.tight_layout(pad=0)
    #                     plt.savefig(os.path.join(plots_folder,x_name+'_'+key+'{k:0>2d}'.format(k=k)+'.png'))
    #                     plt.close()
    #             # x_df = pd.DataFrame(x_array_style)
    #             # x_df.to_csv(os.path.join(plots_folder,x_csv_name))


    groups = [['eps', 'num_classes', 'percentage_labeled'],
              ['eps', 'num_classes', 'percentage_labeled'],
              ['eps', 'num_classes', 'percentage_labeled'],
              ['eps', 'num_classes', 'percentage_labeled'],
              ['eps', 'num_classes', 'percentage_labeled']]

    for sim_id in range(len(constants.results_dir['sbm_sim'])):
        results_file_name = os.path.join(constants.results_dir['sbm_sim'][sim_id], 'comb.json')
        if not Path(results_file_name).is_file():
            warnings.warn(results_file_name + ' not found! Continuing next simulation.')
            continue

        with open(results_file_name) as results_file:
            results = json.load(results_file)


        del results['graph_config']
        # del_keys = []
        # for k, v in results.items():
        #     len_wrong = len(v) not in [1320, 13200]
        #     name_mask = 'tv0_' in k or 'tv00_' in k or 'tv5_' in k or 'tv05_' in k
        #     if len_wrong:
        #         if name_mask:
        #             del_keys.append(k)
        #         else:
        #             print(k, len(v), 'something wrong here')
        # for k in del_keys:
        #     del results[k]
        results_df = pd.DataFrame(results)

        method_names = [col[len('n_err_unlabeled') + 1:] for col in results_df.columns if col.startswith('n_err_unlabeled')]
        pid_list = results_df['pid']
        print('last PID: {p}'.format(p=max(pid_list)))
        print('missing PIDs:')
        print(np.setdiff1d(np.arange(max(pid_list)+1), pid_list))

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
            # csv_file_name += '.csv'
            reduced_csv_file_name = csv_file_name + '.csv'
            subdf = mean_results[mask]
            subdf_reduced = subdf.filter(regex='^(?!num_degenerate)(?!pid)(?!ari)(?!f1)(?!n_err_labeled)(?!n_err_total)(?!acc)\w+')
            subdf_reduced.to_csv(os.path.join(constants.plots_dir['sbm_sim'], reduced_csv_file_name))

            tv_strategies = [n for n in method_names if re.match('tv[0-9]{1,2}_\w?',n) is not None]
            other_strategies = [n for n in method_names if n not in tv_strategies]
            x_min_list = np.unique([float(s[-2:])/100 for s in tv_strategies])
            strategy_list = np.unique([s[:-2] for s in tv_strategies])
            eps_list = np.unique(subdf.reset_index()['eps'])
            # initialize with large negative value to indicate if something went wrong
            n_err = -1000*np.ones((len(x_min_list),len(strategy_list)*len(eps_list)))
            columns = list(itertools.product(strategy_list,eps_list))
            column_names = ['{s}_e{e:0>2d}'.format(s=s,e=int(100*e)) for s, e in columns]
            # x_min_dict = {n: -1000*np.ones((len(x_min_list))) for n in column_names}
            x_min_dict = {**{n: -1000*np.ones((len(x_min_list))) for n in column_names},
                          **{n + '_e{e:0>2d}'.format(e=int(100 * e)): subdf['n_err_unlabeled_'+n][e]*np.ones((len(x_min_list))) for (n, e) in itertools.product(other_strategies, eps_list)},
                          'x_min': x_min_list}
            for i_x, x_min in enumerate(x_min_list):
                for i, (col_name, (strat, eps)) in enumerate(zip(column_names,columns)):
                    x_min_dict[col_name][i_x] = subdf['n_err_unlabeled_'+strat +'{d:0>2d}'.format(d=int(100*x_min))][eps]
            x_min_df = pd.DataFrame.from_dict(x_min_dict).set_index('x_min')
            x_min_csv_file_name = 'x_min_' + csv_file_name + '.csv'
            x_min_df.to_csv(os.path.join(constants.plots_dir['sbm_sim'], x_min_csv_file_name))

            reduced_csv_file_name = 'num_degenerate_' + csv_file_name + '.csv'
            subdf_degenerate = subdf.filter(regex='^(?!pid)(?!ari)(?!f1)(?!n_err)(?!acc)(?!t_run)(?!cut)\w+')
            subdf_degenerate.to_csv(os.path.join(constants.plots_dir['sbm_sim'], reduced_csv_file_name))

        global_cols = groups[sim_id]
        name_dfs = []
        for name in method_names:
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

        if len(groups[sim_id])>2:
            if sim_id == 4:
                for i, df in results_mean.groupby(groups[sim_id][2:]):
                    for nc in [3,5,10]:
                        df_nc = df[df['num_classes']==nc]
                        # df_plot = df_nc[df_nc['name'].isin(['tv2', 'tv3', 'tv4', 'tv5', 'tv_bresson'])]
                        # df_plot = df_nc[~df_nc['name'].str.endswith('fine') & df_nc['name'].str.startswith('tv_regula')]
                        # df_plot = df_nc[df_nc['name'].str.startswith('tv2_re')]
                        # df_plot = df_nc[df_nc['name'].str.startswith('tv_nc') | df_nc['name'].isin(['mapr', 'snc'])]
                        # df_plot = df_nc[df_nc['name'].str.endswith('snc')]
                        df_plot = df_nc
                        plt.figure(figsize=(20, 15))
                        sns.lineplot(data=df_plot, x='eps', y='n_err_unlabeled', hue=groups[sim_id][1], style='name').set(title='n_err: {val}, nc: {nc}'.format(val=i, nc=nc))
                        plt.legend(handlelength=5)
                        plt.show()
                        plt.figure(figsize=(20, 15))
                        sns.lineplot(data=df_plot, x='eps', y='cut', hue=groups[sim_id][1], style='name').set(title='cut: {val}, nc: {nc}'.format(val=i, nc=nc))
                        plt.legend(handlelength=5)
                        plt.show()
                        plt.figure(figsize=(20, 15))
                        sns.lineplot(data=df_plot, x='eps', y='t_run', hue=groups[sim_id][1], style='name').set(title='num_degenerate60: {val}, nc: {nc}'.format(val=i, nc=nc))
                        plt.legend(handlelength=5)
                        plt.show()
        else:
            sns.lineplot(data=results_mean, x='eps', y='n_err_unlabeled', hue=groups[sim_id][1], style='name')
            plt.show()
            sns.lineplot(data=results_mean, x='eps', y='t_run', hue=groups[sim_id][1], style='name')
            plt.show()


def get_graph_config_lists(sim_id):
    eps_list = np.linspace(0, 0.5, 11)
    percentage_labeled_list = [10, 15, 20]
    if sim_id == 0:
        num_classes_list = [2, 3, 5, 10]
        num_nodes_list = [900]*4
    elif sim_id == 1:
        num_classes_list = [2, 3, 5, 10]
        num_nodes_list = [9000]*4
    elif sim_id == 2:
        num_classes_list = [2, 3, 5, 10]
        num_nodes_list = [900]*4
        eps_list = np.linspace(0.3, 0.4, 3)
    elif sim_id == 3:
        num_classes_list = [2, 3, 5, 10]
        num_nodes_list = [9000]*4
        eps_list = np.linspace(0.3, 0.4, 3)
    elif sim_id == 4:
        num_classes_list = [3, 5, 10]
        percentage_labeled_list = [0, 5, 10]
        num_nodes_list = [1200]*3
        eps_list = np.linspace(0, 0.5, 11)
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

    v = 0
    methods = [
        {'name': 'snc', 'method': SpectralLearning(num_classes=num_classes, objective='BNC_INDEF')},
    ]
    if sim_id in [0,1]:
        for e in range(0, 45, 5):
            methods.append({'name': 'tv{e:0>2d}'.format(e=e),
                            'method': TvConvex(num_classes=num_classes, verbosity=v, degenerate_heuristic=None, eps_rel=10**(-e/10), eps_abs=10**(-e/10))})

    if sim_id in [0, 1, 2, 3]:
        if sim_id in [2, 3]:
            x_range = [1, 2, 5, 10, 20, 50, 90]
        else:
            x_range = [5, 10, 50, 90]
        for e in range(10, 35, 5):
            for x in x_range:
                methods.append({'name': 'tv{e:0>2d}_regularization{x:0>2d}'.format(e=e,x=x), 'method': TvConvex(num_classes=num_classes, verbosity=v, degenerate_heuristic='regularize', eps_rel=10**(-e/10), eps_abs=10**(-e/10), regularization_x_min=x/100, return_min_tv=True)})
                methods.append({'name': 'tv{e:0>2d}_resampling{x:0>2d}'.format(e=e,x=x), 'method': TvConvex(num_classes=num_classes, verbosity=v, degenerate_heuristic='rangapuram_resampling', eps_rel=10**(-e/10), eps_abs=10**(-e/10), resampling_x_min=x/100)})

    if sim_id == 4:
        v = 1

        methods.append({'name': 'sponge', 'is_unsupervised': True, 'method': Sponge(num_classes=num_classes)})
        methods.append({'name': 'sncSponge', 'method': SpectralLearning(num_classes=num_classes, objective='SPONGE')})
        methods.append({'name': 'tv15_resampling05', 'method': TvConvex(num_classes=num_classes, verbosity=v, degenerate_heuristic='rangapuram_resampling', eps_rel=10 ** (-15 / 10), eps_abs=10 ** (-15 / 10), resampling_x_min=5 / 100)})
        # methods.append({'name': 'tv_nc1', 'l_guess': 'snc', 'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=1)})
        for b in np.logspace(0,2,5)[1:]:
            for l in [1, 2]:
                for pre in [True, False]:
                    methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_sponge'.format(b=int(10*b),l=l,t=int(pre)),        'l_guess': 'sponge',            'is_unsupervised': True,  'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=b, backtracking_tau_0=0.001, t_max_no_change=None, t_max=1e3)})
                    methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_sncBNC'.format(b=int(10*b),l=l,t=int(pre)),        'l_guess': 'snc',               'is_unsupervised': False, 'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, run_pre_iteration=pre)})
                    methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_sncSponge'.format(b=int(10 * b), l=l, t=int(pre)), 'l_guess': 'sncSponge',         'is_unsupervised': False, 'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, run_pre_iteration=pre)})
                    methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_tvRes'.format(b=int(10*b),l=l,t=int(pre)),         'l_guess': 'tv15_resampling05', 'is_unsupervised': False, 'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, run_pre_iteration=pre)})
                    methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_rand'.format(b=int(10*b),l=l,t=int(pre)),                                          'is_unsupervised': True,  'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, run_pre_iteration=pre)})
                    # methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_tvnc'.format(b=int(10 * b), l=l, t=int(pre)),      'l_guess': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_snc'.format(b=int(10*b),l=l,t=int(pre)),                             'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, run_pre_iteration=pre)})
        # methods.append({'name': 'tv_nc10', 'l_guess': 'tv15_resampling05', 'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=10)})
        # methods.append({'name': 'tv_nc1000', 'l_guess': 'tv15_resampling05', 'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=1000)})
        # methods.append({'name': 'tv_nc10000', 'l_guess': 'tv15_resampling05', 'method': TvNonConvex(num_classes=num_classes, verbosity=v, penalty_parameter=10000)})

    if sim_id > len(constants.results_dir['sbm_sim']):
        raise ValueError('unknown sim_id')

    if sim_id == 0:
        methods.append({'name': 'mapr', 'l_guess': 'snc', 'method': LsbmMap(num_classes=num_classes, verbosity=v, pi=pi, pe=pe, li=li, le=le, class_distribution=class_distribution, eps=1e-3)},)

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
    sim.save_results(constants.results_dir['sbm_sim'][sim_id], split_file=False, save_degenerate_stats=False)

    if pid in [24, 34]:#< len(sim.graph_config_list):
        filename = os.path.join(constants.plots_dir['sbm_sim'],
                                'x_s{sid}_p{pid}.json'.format(sid=sim_id, pid=pid))
        x_lists = {k: v.tolist() for k, v in sim.embedding.items() if k in ['tv{e:0>2d}'.format(e=e) for e in range(0,45,5)]}
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
            for i in range(len(constants.results_dir['sbm_sim'])):
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
