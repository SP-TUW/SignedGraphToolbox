import os

results_dir = {'global': os.path.join('results', 'TV')}
plots_dir = {'global': os.path.join(results_dir['global'], 'plots')}

results_dir['sbm_sim'] = [os.path.join(results_dir['global'], 'sbm_sim_{sid}'.format(sid=sim_id)) for sim_id in range(10)]

plots_dir['sbm_sim'] = os.path.join(plots_dir['global'], 'sbm_sim')