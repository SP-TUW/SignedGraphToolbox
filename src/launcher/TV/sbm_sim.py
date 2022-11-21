import json
import os

import numpy as np

from src.launcher import SBMSimulation
from src.launcher.TV import constants, plotting
from src.node_classification import SpectralLearning, Sponge, TvAugmentedADMM, TvStandardADMM, LsbmMap, DiffuseInterface


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


def get_graph_config_lists(sim_id, return_name=False):
    name = ''
    eps_list = np.linspace(0, 0.5, 11)
    percentage_labeled_list = [10/3, 10, 20]
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
        name = 'balancedness sweep with several values for beta'
        num_classes_list = [3, 5, 10]
        percentage_labeled_list = [0, 10/3, 10, 20]
        num_nodes_list = [900]*3
        eps_list = np.linspace(0, 0.5, 11)
    elif sim_id == 5:
        name = 'simulations from Asilomar'
        num_classes_list = [3]
        percentage_labeled_list = [0]
        num_nodes_list = [900]
        eps_list = np.linspace(0.0, 0.5*(1-1/100), 100)
    elif sim_id == 6:
        name = 'high rep simulations for small range of balancedness with denser steps'
        num_classes_list = [3, 5, 10]
        percentage_labeled_list = [0, 10/3, 10, 20]
        num_nodes_list = [900]*3
        eps_list = np.linspace(0.15, 0.35, 9)
    elif sim_id == 7:
        name = 'high rep simulations for varying beta'
        num_classes_list = [3, 5, 10]
        percentage_labeled_list = [0, 10/3, 10, 20]
        num_nodes_list = [900]*3
        eps_list = np.linspace(0.15, 0.35, 5)
    elif sim_id == 8:
        name = 'balancedness sweep for nonconvex augmented admm'
        num_classes_list = [3, 5, 10]
        percentage_labeled_list = [0, 10/3, 10, 20]
        num_nodes_list = [900]*3
        eps_list = np.linspace(0, 0.5, 11)
    elif sim_id == 9:
        name = 'balancedness sweep for high repetition comparison of all algorithms'
        num_classes_list = [3, 5, 10]
        percentage_labeled_list = [0, 10/3, 10, 20]
        num_nodes_list = [900]*3
        eps_list = np.linspace(0.0, 0.5, 11)
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
    if return_name:
        return config_lists, name
    else:
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
        {'name': 'sncBNC', 'method': SpectralLearning(num_classes=num_classes, objective='BNC_INDEF')},
        {'name': 'sncSponge', 'method': SpectralLearning(num_classes=num_classes, objective='SPONGE')},
    ]
    if sim_id in [0,1]:
        for e in range(0, 45, 5):
            methods.append({'name': 'tv{e:0>2d}'.format(e=e),
                            'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, degenerate_heuristic=None, eps_rel=10 ** (-e / 10), eps_abs=10 ** (-e / 10))})

    if sim_id in [0, 1, 2, 3]:
        if sim_id in [2, 3]:
            x_range = [1, 2, 5, 10, 20, 50, 90]
        else:
            x_range = [5, 10, 50, 90]
        for e in range(10, 35, 5):
            for x in x_range:
                methods.append({'name': 'tv{e:0>2d}_regularization{x:0>2d}'.format(e=e,x=x), 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, degenerate_heuristic='regularize', eps_rel=10 ** (-e / 10), eps_abs=10 ** (-e / 10), regularization_x_min=x / 100, return_min_tv=True)})
                methods.append({'name': 'tv{e:0>2d}_resampling{x:0>2d}'.format(e=e,x=x), 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, degenerate_heuristic='rangapuram_resampling', eps_rel=10 ** (-e / 10), eps_abs=10 ** (-e / 10), resampling_x_min=x / 100)})

    if sim_id == 4:
        v = 1
        # methods.append({'name': 'sponge', 'is_unsupervised': True, 'method': Sponge(num_classes=num_classes)})
        methods.append({'name': 'tv15_resampling05', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, degenerate_heuristic='rangapuram_resampling', eps_rel=10 ** (-15 / 10), eps_abs=10 ** (-15 / 10), resampling_x_min=5 / 100)})
        methods.append({'name': 'tv_nc_beta+2_pre2_rand_asilomar',                           'is_unsupervised': False, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=100, pre_iteration_version=2, t_max=1000, eps=1e-3 / np.sqrt(900 * 3), t_max_no_change=None, eps_inner=1e-3, t_max_inner=1000, backtracking_param=0, backtracking_tau_0=0.01)})
        methods.append({'name': 'tv_nc_beta+2_pre2_sponge_asilomar', 'l_guess': 'sncSponge',    'is_unsupervised': False, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=100, pre_iteration_version=2, t_max=1000, eps=1e-3 / np.sqrt(900 * 3), t_max_no_change=None, eps_inner=1e-3, t_max_inner=1000, backtracking_param=0, backtracking_tau_0=0.01)})
        for b in np.logspace(0,5,6):
            for pre in [0, 1, 2]:
                # methods.append({'name': 'tv_nc_beta{b:0>+1d}_pre{t}_sponge'.format(b=int(b),t=int(pre)),      'l_guess': 'sponge',            'is_unsupervised': True,  'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, pre_iteration_version=pre, t_max_no_change=None)})
                for l_guess in ['sncSponge']:
                    methods.append({'name': 'tv_nc_beta{b:0>+1d}_pre{t}_{g}'.format(b=int(np.log10(b)),t=int(pre), g=l_guess), 'l_guess': l_guess, 'is_unsupervised': False, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, pre_iteration_version=pre, t_max_no_change=None)})
                methods.append({'name': 'tv_nc_beta{b:0>+1d}_pre{t}_rand'.format(b=int(np.log10(b)),t=int(pre)),                                   'is_unsupervised': False,  'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, pre_iteration_version=pre, t_max_no_change=None)})
                # methods.append({'name': 'tv_nc_beta{b:0>+1d}_pre{t}_tvnc'.format(b=int(10 * b), t=int(pre)),      'l_guess': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_snc'.format(b=int(10*b),l=l,t=int(pre)),                             'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, run_pre_iteration=pre)})


    if sim_id == 5:
        v = 1
        methods = [
            {'name': 'sponge', 'is_unsupervised': True, 'method': Sponge(num_classes=num_classes)},
            {'name': 'spongeSym', 'is_unsupervised': True, 'method': Sponge(num_classes=num_classes,objective='SYM')},
            ]
        for b in [100]:#np.logspace(1,2,2):
            for l in [1]:
                pre = 2
                methods.append({'name': 'tv_nc_beta{b:0>5d}_l{l:d}_pre{t}_rand'.format(b=int(b),l=l,t=int(pre)),                                 'is_unsupervised': True, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, pre_iteration_version=pre, t_max=1000, eps=1e-3 / np.sqrt(900 * 3), t_max_no_change=None, eps_inner=1e-3, t_max_inner=1000, backtracking_param=0, backtracking_tau_0=0.01)})
                methods.append({'name': 'tv_nc_beta{b:0>5d}_l{l:d}_pre{t}_sponge'.format(b=int(b),l=l,t=int(pre)),       'l_guess': 'sponge',    'is_unsupervised': True, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, pre_iteration_version=pre, t_max=1000, eps=1e-3 / np.sqrt(900 * 3), t_max_no_change=None, eps_inner=1e-3, t_max_inner=1000, backtracking_param=0, backtracking_tau_0=0.01)})

                # pre = 1
                # methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_rand'.format(b=int(b),l=l,t=int(pre)),                                 'is_unsupervised': True, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, pre_iteration_version=pre, t_max=10000, eps=1e-3, t_max_no_change=None, eps_inner=1e-3, t_max_inner=1000, backtracking_param=0, backtracking_tau_0=0.01)})
                # methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_sponge'.format(b=int(b),l=l,t=int(pre)),       'l_guess': 'sponge',    'is_unsupervised': True, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, pre_iteration_version=pre, t_max=10000, eps=1e-3, t_max_no_change=None, eps_inner=1e-3, t_max_inner=1000, backtracking_param=0, backtracking_tau_0=0.01)})

                pre = 0
                methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_rand'.format(b=int(b),l=l,t=int(pre)),                                 'is_unsupervised': True, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, pre_iteration_version=pre, t_max=10000, eps=1e-3, eps_admm=1e-5, t_max_no_change=None, eps_inner=1e-8, t_max_inner=10000, backtracking_param=1 / 2, backtracking_tau_0=0.01)})
                methods.append({'name': 'tv_nc_beta{b:0>4d}_l{l:d}_pre{t}_sponge'.format(b=int(b),l=l,t=int(pre)),       'l_guess': 'sponge',    'is_unsupervised': True, 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, laplacian_scaling=l, pre_iteration_version=pre, t_max=10000, eps=1e-3, eps_admm=1e-5, t_max_no_change=None, eps_inner=1e-8, t_max_inner=10000, backtracking_param=1 / 2, backtracking_tau_0=0.01)})

    if sim_id == 6:
        # high repetition nonconvex TV
        v = 1
        # methods.append({'name': 'sponge', 'is_unsupervised': True, 'method': Sponge(num_classes=num_classes)})
        methods.append({'name': 'tv15_resampling05', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v,
                                                                               degenerate_heuristic='rangapuram_resampling',
                                                                               eps_rel=10 ** (-15 / 10),
                                                                               eps_abs=10 ** (-15 / 10),
                                                                               resampling_x_min=5 / 100)})
        for b in np.logspace(3, 5, 3):
            for pre in [0]:
                # methods.append({'name': 'tv_nc_beta{b:0>+1d}_pre{t}_sponge'.format(b=int(b),t=int(pre)),      'l_guess': 'sponge',            'is_unsupervised': True,  'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, pre_iteration_version=pre, t_max_no_change=None)})
                for l_guess in ['sncSponge']:
                    methods.append(
                        {'name': 'tv_nc_beta{b:0>+1d}_pre{t}_{g}'.format(b=int(np.log10(b)), t=int(pre), g=l_guess),
                         'l_guess': l_guess, 'is_unsupervised': False,
                         'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b,
                                                  pre_iteration_version=pre, t_max_no_change=None)})

    if sim_id > len(constants.results_dir['sbm_sim']):
        raise ValueError('unknown sim_id')

    if sim_id == 7:
        # high repetition nonconvex TV
        v = 1
        for b in np.logspace(0, 5, 11):
            for pre in [0]:
                for l_guess in ['sncSponge']:
                    methods.append(
                        {'name': 'tv_nc_beta{b:0>+1.1f}_pre{t}_{g}'.format(b=np.log10(b), t=int(pre), g=l_guess),
                         'l_guess': l_guess, 'is_unsupervised': False,
                         'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b,
                                                  pre_iteration_version=pre, t_max_no_change=None)})


    if sim_id == 8:
        # high repetition nonconvex TV
        v = 1
        methods.append({'name': 'tv15_resampling05', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v,
                                                                               degenerate_heuristic='rangapuram_resampling',
                                                                               eps_rel=10 ** (-15 / 10),
                                                                               eps_abs=10 ** (-15 / 10),
                                                                               resampling_x_min=5 / 100)})
        for b in np.logspace(0, 5, 6):
            for l_guess in ['sncSponge']:
                methods.append(
                    {'name': 'tv_nc_beta{b:0>+1.1f}_{g}'.format(b=np.log10(b), g=l_guess),
                     'l_guess': l_guess, 'is_unsupervised': False,
                     'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, penalty_strat_threshold=float('inf'), y0=0, y1=0, min_norm=num_classes-2)})
            methods.append(
                    {'name': 'tv_nc_beta{b:0>+1.1f}_rand'.format(b=np.log10(b)),
                     'is_unsupervised': False,
                     'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b, penalty_strat_threshold=float('inf'), y0=0, y1=0, min_norm=num_classes-2)})

    if sim_id == 9:
        # high repetition nonconvex TV
        v = 1
        for num_eig in [10, 20, 50, 100]:
            methods.append({'name': 'diffuseInterface_sym{n:0>3d}'.format(n=num_eig), 'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='sym', num_eig=num_eig)})
            methods.append({'name': 'diffuseInterface_am{n:0>3d}'.format(n=num_eig), 'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='am', num_eig=num_eig)})
            methods.append({'name': 'diffuseInterface_lap{n:0>3d}'.format(n=num_eig), 'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='lap', num_eig=num_eig)})
            methods.append({'name': 'diffuseInterface_sponge{n:0>3d}'.format(n=num_eig), 'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='sponge', num_eig=num_eig)})
        methods.append({'name': 'tv15_resampling05', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v,
                                                                               degenerate_heuristic='rangapuram_resampling',
                                                                               eps_rel=10 ** (-15 / 10),
                                                                               eps_abs=10 ** (-15 / 10),
                                                                               resampling_x_min=5 / 100)})
        b = 1e4
        pre = 0
        l_guess = 'sncSponge'
        methods.append({'name': 'tv_nc_beta{penalty:0>+1.1f}_pre{pre}_{guess}'.format(penalty=np.log10(b), pre=int(pre), guess=l_guess),
                        'l_guess': l_guess, 'is_unsupervised': False,
                        'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b,
                                                  pre_iteration_version=pre, t_max_no_change=None)})

    if sim_id > len(constants.results_dir['sbm_sim']):
        raise ValueError('unknown sim_id')

    if sim_id in [0,4,6,8]:
        methods.append({'name': 'maprSNC', 'l_guess': 'sncSponge', 'method': LsbmMap(num_classes=num_classes, verbosity=v, pi=pi, pe=pe, li=li, le=le, class_distribution=class_distribution, eps=1e-3)},)
        methods.append({'name': 'maprMinErr', 'l_guess': 'min_err', 'method': LsbmMap(num_classes=num_classes, verbosity=v, pi=pi, pe=pe, li=li, le=le, class_distribution=class_distribution, eps=1e-3)},)
        methods.append({'name': 'maprMinCut', 'l_guess': 'min_cut', 'method': LsbmMap(num_classes=num_classes, verbosity=v, pi=pi, pe=pe, li=li, le=le, class_distribution=class_distribution, eps=1e-3)},)

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
    sim.save_results(constants.results_dir['sbm_sim'][sim_id], split_file=False, save_degenerate_stats=False, reduce_data=True)

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
                config_lists, name = get_graph_config_lists(sim_id=i, return_name=True)
                sim = SBMSimulation(**config_lists)
                print('{n: >3d} configs in simulation {i} --- {name}'.format(n=len(sim.graph_config_list), i=i, name=name))
        elif args[1] == '-p':

            plotting.plot()
        else:
            print('unknown command {c}'.format(c=args[1]))
        pass
    else:
        pid, sim_id = args_to_pid_and_sim_id(sys.argv)
        run(pid, sim_id)
