import os

results_dir = {'global': os.path.join('results', 'LSBM_MAP')}
plots_dir = {'global': os.path.join(results_dir['global'], 'plots')}

results_dir['sbm_sim'] = [os.path.join(results_dir['global'], 'sbm_sim_lelarge'),
                          os.path.join(results_dir['global'], 'sbm_sim_hajek'),
                          os.path.join(results_dir['global'], 'sbm_sim_yun')]

plots_dir['sbm_sim'] = os.path.join(plots_dir['global'], 'sbm_sim')