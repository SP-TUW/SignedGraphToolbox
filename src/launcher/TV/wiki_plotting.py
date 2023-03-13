def plot():
    import json
    import os
    import pandas as pd
    import numpy as np
    import re
    from src.launcher.TV import constants

    # % HF(ours) & 0.632 & 0.847 & 0.854 & 0.858 & 0.615 & 0.624 & 0.634 & 0.642 & 0.557 & 0.579 & 0.603 & 0.622

    result_dfs = {}
    keys = ['cut','acc_unlabeled','acc_total','t_run']

    for sim_id in range(6):
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
            filtered_df = perc_lab_df.filter(regex='(?={k})'.format(k=key),axis=1)
            if sim_id < 3:
                filtered_df = filtered_df.filter(regex='{k}(?!_HF)'.format(k=key), axis=1).filter(regex='{k}(?!_gt)'.format(k=key), axis=1)
            filtered_df.columns = [c.replace(key+'_','').replace('_','') for c in filtered_df.columns]
            filtered_df = filtered_df.unstack()
            result_dfs[key].setdefault(name, None)
            # result_dfs[key][name] = filtered_df
            result_dfs[key][name] = pd.concat((result_dfs[key][name],filtered_df),axis=0)
            # print(result_dfs[key][name])

    for key in keys:
        joint_df = pd.DataFrame(result_dfs[key])
        joint_df.columns = [c.replace('_','-') for c in joint_df.columns]
        joint_df.columns = [c.replace('WIKI','Wikipedia') for c in joint_df.columns]
        joint_df.columns = [c.replace('EDITOR','Editor') for c in joint_df.columns]
        joint_df.columns = [c.replace('ELEC','Elections') for c in joint_df.columns]
        joint_df.columns = [c.replace('RFA','RfA') for c in joint_df.columns]
        joint_df = joint_df.unstack('percentage_labeled')
        joint_df.columns.names = ('','$M$')
        # joint_df = joint_df.drop('DIam200',axis=0)
        # joint_df = joint_df.drop('DIsym200',axis=0)
        joint_df = joint_df.drop('tv10',axis=0)
        joint_df = joint_df.drop('tv10reg90',axis=0)
        joint_df = joint_df.drop('tv10res',axis=0)
        joint_df = joint_df.drop('HF',axis=0)
        joint_df = joint_df.drop('sncAM',axis=0)
        joint_df = joint_df.drop('sncBNC',axis=0)
        joint_df = joint_df.drop('sncBNCIndef',axis=0)
        joint_df = joint_df.drop('sncNC',axis=0)
        joint_df = joint_df.drop('sncRC',axis=0)
        if 'gt' in joint_df.index:
            joint_df = joint_df.drop('gt',axis=0)

        names = [n[:4] for n in joint_df.index]
        i_diff = np.flatnonzero([names[i] != names[i-1] for i in range(1,len(names))])+1

        joint_df.index = [re.sub('DI([a-z]{2,3})[0]?([0-9]{2,3})','$\\\\mathrm{DI}(\\\\mt{L}_{\\\\mathrm{\\1}}^{\\2})$',n) for n in joint_df.index]
        joint_df.index = [re.sub('tv([0-9]{2})reg90','$\\\\mathrm{TV}_{\\\\mathrm{REG}}^{\\1}$',n) for n in joint_df.index]
        joint_df.index = [re.sub('tv([0-9]{2})res','$\\\\mathrm{TV}_{\\\\mathrm{RES}}^{\\1}$',n) for n in joint_df.index]
        joint_df.index = [re.sub('tv([0-9]{2})','$\\\\mathrm{TV}{\\1}$',n) for n in joint_df.index]
        # joint_df.index = [re.sub('tv([0-9]{2})reg90','$\\\\mathrm{TV}\\1$',n) for n in joint_df.index]

        joint_df.index = [n if i not in i_diff else '\\hline{}'+n for i,n in enumerate(joint_df.index)]

        print(joint_df.to_string())
        if key=='cut':
            scaled_df = (joint_df/1000)
            r = np.zeros(3)
            for i_sim in range(3):
                i_sim_df = scaled_df.iloc[:,4*i_sim:4*(i_sim+1)]
                i_sim_min = i_sim_df.min(axis=0)
                diff_df = i_sim_df - i_sim_min
                diff_max = diff_df.max(axis=1)
                r[i_sim] = np.sort(diff_max)[0]
            s = scaled_df.style.highlight_min(props='textbf:--rwrap', axis=0).format(precision=1)
            s = s.highlight_between(axis=1, left=scaled_df.min(axis=0)+np.array([r[0]]*4+[r[1]]*4+[r[2]]*4),right=scaled_df.max(axis=0), inclusive='right', props='color{lightgray}:--rwrap')
        else:
            scaled_df = joint_df
            r = np.zeros(3)
            for i_sim in range(3):
                i_sim_df = scaled_df.iloc[:,4*i_sim:4*(i_sim+1)]
                i_sim_max = i_sim_df.max(axis=0)
                diff_df = i_sim_max - i_sim_df
                diff_max = diff_df.max(axis=1)
                r[i_sim] = np.sort(diff_max)[2]
            s = scaled_df.style.highlight_max(props='textbf:--rwrap', axis=0).format(precision=3)
            s = s.highlight_between(axis=1, right=scaled_df.max(axis=0)-np.array([r[0]]*4+[r[1]]*4+[r[2]]*4),left=scaled_df.min(axis=0), inclusive='left', props='color{lightgray}:--rwrap')
            # s = s.highlight_quantile(axis=0, q_right=1-5/scaled_df.shape[0], inclusive='both', props='color{lightgray}:--rwrap')
        # s.render

        thesis_path = os.path.join('~','Desktop','LatexRepositories','PHD-Thesis')
        thesis_path = os.path.join('~','Desktop','LatexRepositories','Publications','TVConvex')
        s.to_latex(buf=os.path.join(thesis_path,'source','figures','TV_CONVEX', 'wiki_sim', '{k}.tex'.format(k=key)),hrules=True,multicol_align='r',column_format='l||r|r|r|r||r|r|r|r||r|r|r|r')
        s.to_html(buf=os.path.join(thesis_path,'source','figures','TV_CONVEX', 'wiki_sim', '{k}.html'.format(k=key)))

    # print('HF(ours)              0.632     0.847     0.854     0.858     0.615     0.624     0.634     0.642     0.557     0.579     0.603     0.622')

    pass

#
#                    WIKI_EDITOR                               WIKI_ELEC                                WIKI_RFA
# percentage_labeled          1         5         10        15        1         5         10        15        1         5         10       15
# acc_unlabeled_HF      0.664704  0.788935  0.814968  0.828462  0.564646  0.580211  0.587508  0.594873  0.546531  0.585752  0.601616  0.60853
#                    WIKI_EDITOR                               WIKI_ELEC                               WIKI_RFA
# percentage_labeled          1         5         10        15        1         5         10        15       1       5         10        15
# acc_total_HF          0.668029  0.799472  0.833453  0.854183  0.568788  0.601022  0.628613  0.655328  0.55088  0.6063  0.641402  0.667173