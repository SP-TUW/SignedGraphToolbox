def plot():
    import json
    import os
    import pandas as pd
    from src.launcher.TV import constants

    # % HF(ours) & 0.632 & 0.847 & 0.854 & 0.858 & 0.615 & 0.624 & 0.634 & 0.642 & 0.557 & 0.579 & 0.603 & 0.622

    result_dfs = {}
    keys = ['acc_unlabeled']

    for sim_id in range(3,6):
        results_file_name = os.path.join(constants.results_dir['wiki_sim'][sim_id], 'comb.json')
        with open(results_file_name) as results_file:
            results = json.load(results_file)
        name = results['graph_name'][0][:-2]
        print(name, results['num_nodes'][0])
        results_df = pd.DataFrame(results)
        print(results_df.columns)
        results_df = results_df.drop(['graph_config'],axis=1)
        # print(results_df[['percentage_labeled','acc_unlabeled_HF','graph_name']].groupby(['percentage_labeled','graph_name']).mean())
        # results_df = results_df[results_df['pid']<40]
        # perc_lab_df = results_df.groupby('percentage_labeled').head(100).groupby('percentage_labeled').mean()
        perc_lab_df = results_df.groupby('percentage_labeled').mean()
        # print(perc_lab_df)
        perc_lab_df = perc_lab_df.drop(['pid','num_nodes'],axis=1)
        for key in keys:
            result_dfs.setdefault(key,{})
            result_dfs[key][name] = perc_lab_df.filter(regex='(?={k})'.format(k=key),axis=1).unstack()

    for key in keys:
        joint_df = pd.DataFrame(result_dfs[key]).unstack('percentage_labeled')
        print(joint_df.to_string())

    print('HF(ours)              0.632     0.847     0.854     0.858     0.615     0.624     0.634     0.642     0.557     0.579     0.603     0.622')

    pass

#
#                    WIKI_EDITOR                               WIKI_ELEC                                WIKI_RFA
# percentage_labeled          1         5         10        15        1         5         10        15        1         5         10       15
# acc_unlabeled_HF      0.664704  0.788935  0.814968  0.828462  0.564646  0.580211  0.587508  0.594873  0.546531  0.585752  0.601616  0.60853
#                    WIKI_EDITOR                               WIKI_ELEC                               WIKI_RFA
# percentage_labeled          1         5         10        15        1         5         10        15       1       5         10        15
# acc_total_HF          0.668029  0.799472  0.833453  0.854183  0.568788  0.601022  0.628613  0.655328  0.55088  0.6063  0.641402  0.667173