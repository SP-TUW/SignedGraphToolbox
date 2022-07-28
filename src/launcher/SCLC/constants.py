import os

results_dir = {'global': os.path.join('results', 'DSLW2022')}
plots_dir = {'global': os.path.join(results_dir['global'], 'plots')}

results_dir['sbm_sim'] = os.path.join(results_dir['global'], 'sbm_sim')
results_dir['init_sim'] = os.path.join(results_dir['global'], 'init_sim')
results_dir['real_world'] = os.path.join(results_dir['global'], 'real_world')

plots_dir['sbm_sim'] = os.path.join(plots_dir['global'], 'sbm_sim')
plots_dir['init_sim'] = os.path.join(plots_dir['global'], 'init_sim')
plots_dir['real_world'] = os.path.join(plots_dir['global'], 'real_world')