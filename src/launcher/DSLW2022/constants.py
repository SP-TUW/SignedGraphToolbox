import os

results_dir = {'global': os.path.join('results', 'DSLW2022')}
plots_dir = {'global': os.path.join(results_dir['global'], 'plots')}

results_dir['sbm_curve'] = os.path.join(results_dir['global'], 'sbm_curve')

plots_dir['sbm_curve'] = os.path.join(plots_dir['global'], 'sbm_curve')