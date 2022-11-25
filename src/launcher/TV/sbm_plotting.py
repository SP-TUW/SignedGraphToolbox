def plot():
    print('plotting')
    import itertools
    import json
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    import seaborn as sns
    import warnings
    from pathlib import Path
    from src.launcher.TV import constants




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


    groups = [['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled'],
              ['stopping_tol', 'num_classes', 'percentage_labeled']]

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
        results_df.columns = results_df.columns.str.replace("+", "p")
        results_df.columns = results_df.columns.str.replace("-", "m")
        results_df.columns = results_df.columns.str.replace(".", "")

        method_names = [col[len('n_err_unlabeled') + 1:] for col in results_df.columns if col.startswith('n_err_unlabeled')]
        pid_list = results_df['pid']
        print('last PID: {p}'.format(p=max(pid_list)))
        print('missing PIDs:')
        print(np.setdiff1d(np.arange(max(pid_list)+1), pid_list))

        if sim_id in [7]:
            for c in results_df.columns:
                if c.startswith('t_run'):
                    name = c[5:]
                    # results_df[c] /= results_df['t_run_sncSponge']
                    if 't_run_norm_tv_nc_betap5_pre0_sncSponge' in results_df.columns:
                        results_df['t_run_norm' + name] = results_df[c] / results_df['t_run_norm_tv_nc_betap5_pre0_sncSponge']
                    else:
                        results_df['t_run_norm' + name] = results_df[c] / results_df['t_run_norm_tv_nc_betap50_pre0_sncSponge']

        mean_results = results_df.groupby(groups[sim_id]).mean().reset_index(level=list(range(1, len(groups[sim_id]))))
        unique_lists = []
        for grouping in groups[sim_id][1:]:
            unique_lists.append(results_df[grouping].unique())
        for ind in itertools.product(*unique_lists):
            mask = np.ones(mean_results.shape[0], dtype=bool)
            csv_file_name = 'sbm_mean_sim_{sid}'.format(sid=sim_id)
            for key, val in zip(groups[sim_id][1:], ind):
                mask = np.bitwise_and(mask, mean_results[key] == val)
                csv_file_name += '_{k}_{v}'.format(k=key, v=np.round(val,2))
            # csv_file_name += '.csv'
            reduced_csv_file_name = csv_file_name + '.csv'
            subdf = mean_results[mask]
            subdf_reduced = subdf.filter(regex='^(?!num_degenerate)(?!pid)(?!ari)(?!f1)(?!n_err_labeled)(?!n_err_total)(?!acc)\w+')
            subdf_reduced.to_csv(os.path.join(constants.plots_dir['sbm_sim'], reduced_csv_file_name))

            tv_strategies = [n for n in method_names if re.match('tv[0-9]{1,2}_\w?',n) is not None]
            other_strategies = [n for n in method_names if n not in tv_strategies]
            x_min_list = np.unique([float(s[-2:])/100 for s in tv_strategies])
            strategy_list = np.unique([s[:-2] for s in tv_strategies])
            eps_list = np.unique(subdf.reset_index()['stopping_tol'])
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
        results_max = results_df.groupby(['name'] + groups[sim_id]).max().reset_index()

        if len(groups[sim_id]) > 2:
            if sim_id == -1:
                for i, df in results_mean.groupby(groups[sim_id][2:]):
                    for nc in [3, 5, 10]:
                        df_nc = df[df['num_classes']==nc]
                        # df_plot = df_nc[df_nc['name'].isin(['tv2', 'tv3', 'tv4', 'tv5', 'tv_bresson'])]
                        # df_plot = df_nc[~df_nc['name'].str.endswith('fine') & df_nc['name'].str.startswith('tv_regula')]
                        # df_plot = df_nc[df_nc['name'].str.startswith('tv2_re')]
                        # df_plot = df_nc[df_nc['name'].str.startswith('tv_nc') | df_nc['name'].isin(['mapr', 'snc'])]
                        # df_plot = df_nc[df_nc['name'].str.endswith('snc')]
                        df_plot = df_nc
                        # df_plot = df_nc[df_nc['name'].str.startswith('tv_nc_beta0100_l1_pre0') | df_nc['name'].str.startswith('sponge')]
                        # df_plot = df_nc[df_nc['name'].str.endswith('sncSponge') | df_nc['name'].str.startswith('sncSponge') | df_nc['name'].str.startswith('mapr')]

                        plt.figure(figsize=(20, 15))
                        p1 = sns.lineplot(data=df_plot, x='stopping_tol', y='n_err_unlabeled', hue=groups[sim_id][1], style='name')
                        p1.set(title='n_err at pl: {val}, nc: {nc}'.format(val=i, nc=nc))
                        # p1.set(yscale='log')
                        plt.grid()
                        plt.legend(handlelength=5)
                        plt.show()

                        # plt.figure(figsize=(20, 15))
                        # sns.lineplot(data=df_plot, x='stopping_tol', y='cut', hue=groups[sim_id][1], style='name').set(title='cut at pl: {val}, nc: {nc}'.format(val=i, nc=nc))
                        # plt.grid()
                        # plt.legend(handlelength=5)
                        # plt.show()

                        # plt.figure(figsize=(20, 15))
                        # sns.lineplot(data=df_plot, x='stopping_tol', y='t_run', hue=groups[sim_id][1], style='name').set(title='t_run at pl: {val}, nc: {nc}'.format(val=i, nc=nc))
                        # plt.grid()
                        # plt.legend(handlelength=5)
                        # plt.show()
        else:
            sns.lineplot(data=results_mean, x='stopping_tol', y='n_err_unlabeled', hue=groups[sim_id][1], style='name')
            plt.show()
            sns.lineplot(data=results_mean, x='stopping_tol', y='t_run', hue=groups[sim_id][1], style='name')
            plt.show()

