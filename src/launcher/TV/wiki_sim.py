from src.launcher.TV import constants
from src.launcher import ClassificationSimulation
from src.launcher.TV import wiki_plotting as plotting
from src.node_classification import DiffuseInterface, HarmonicFunctions, SpectralLearning, TvAugmentedADMM

from src.graphs import graph_factory

import numpy as np


def make_result_dirs():
    print('making result and plot directories')
    from pathlib import Path
    for dir in constants.results_dir['wiki_sim']:
        Path(dir).mkdir(parents=True, exist_ok=True)
    Path(constants.plots_dir['wiki_sim']).mkdir(parents=True, exist_ok=True)


def combine_results():
    print('combining results')
    from src.tools.combine_results import combine_results as cr
    for dir in constants.results_dir['wiki_sim']:
        cr(dir, has_lists=False)


def get_graph_config_lists(sim_id, return_name=False):
    config_lists = []
    graph_args = {}
    if sim_id == 0:  # wiki editor
        name = 'WIKI_EDITOR'
    elif sim_id == 1:  # wiki elec
        name = 'WIKI_ELEC'
        graph_args['combination_method'] = 'mean_sign'
        graph_args['from_matlab'] = True
    elif sim_id == 2:  # wiki RfA
        name = 'WIKI_RFA'
        graph_args['combination_method'] = 'mean_sign'
        graph_args['from_matlab'] = True
    elif sim_id == 3:  # wiki editor
        name = 'WIKI_EDITOR'
        graph_args = {'only_pos': True}
    elif sim_id == 4:  # wiki elec
        name = 'WIKI_ELEC'
        graph_args['combination_method'] = 'only_pos'
        graph_args['from_matlab'] = False
    elif sim_id == 5:  # wiki RfA
        name = 'WIKI_RFA'
        graph_args['combination_method'] = 'only_pos'
        graph_args['from_matlab'] = False
    else:
        raise ValueError('unknown sim_id')

    for p in [1, 5, 10, 15]:
        config_lists.append({'model': name, 'percentage_labeled': p, 'is_percentage': True, 'name': '{n}{p:0>2d}'.format(n=name,p=p), **graph_args})

    if return_name:
        return config_lists, name
    else:
        return config_lists


def get_methods(graph_config, sim_id):
    graph = graph_factory.make_graph(**graph_config)
    class_prior = np.mean(graph.class_labels)
    num_classes = 2
    v = 1
    if sim_id >= 3:
        methods = [{'name': 'HF', 'method': HarmonicFunctions(num_classes=num_classes, class_prior=class_prior)}]
    else:
        methods = [{'name': 'HF', 'method': HarmonicFunctions(num_classes=num_classes, class_prior=class_prior)},
                   {'name': 'sncAM', 'method': SpectralLearning(num_classes=num_classes, objective='AM')},
                   {'name': 'sncRC', 'method': SpectralLearning(num_classes=num_classes, objective='RC')},
                   {'name': 'sncNC', 'method': SpectralLearning(num_classes=num_classes, objective='NC')},
                   {'name': 'sncBNC', 'method': SpectralLearning(num_classes=num_classes, objective='BNC')},
                   {'name': 'sncBNCIndef', 'method': SpectralLearning(num_classes=num_classes, objective='BNC_INDEF')}
                    ]
        for e in range(10,45,5):
            methods.append({'name': 'tv{e:2d}'.format(e=e), 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, degenerate_heuristic=None, eps_rel=10 ** (-e / 10), eps_abs=10 ** (-e / 10), resampling_x_min=90 / 100)})
            methods.append({'name': 'tv{e:2d}_res'.format(e=e), 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, degenerate_heuristic='rangapuram_resampling', eps_rel=10 ** (-e / 10), eps_abs=10 ** (-e / 10), resampling_x_min=5 / 100)})
            methods.append({'name': 'tv{e:2d}_reg90'.format(e=e), 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v, degenerate_heuristic='regularize', eps_rel=10 ** (-e / 10), eps_abs=10 ** (-e / 10), regularization_x_min=90 / 100, regularization_max=2 ** 15, return_min_tv=True)})
        # b = 1e4
        # pre = 0
        # l_guess = 'sncSponge'
        # methods.append({'name': 'tv_nc_beta{penalty:0>+1.1f}_pre{pre}_{guess}'.format(penalty=np.log10(b), pre=int(pre),
        #                                                                               guess=l_guess),
        #                 'l_guess': l_guess, 'is_unsupervised': False,
        #                 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b,
        #                                          pre_iteration_version=pre, t_max_no_change=None)})

        if graph_config['model'] in ['WIKI_ELEC', 'WIKI_RFA']:
            # num_eig_list = [20,40,60,80,100]
            num_eig_list = [20, 100, 200]
            use_full_matrix = False
        else:
            # num_eig_list = [200,400,600,800,1000]
            num_eig_list = [20, 100, 200]
            use_full_matrix = False

        for num_eig in num_eig_list:
            methods.append({'name': 'DI_sym{n:0>3d}'.format(n=num_eig),
                            'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='sym',
                                                       num_eig=num_eig, use_full_matrix=use_full_matrix)})
            methods.append({'name': 'DI_am{n:0>3d}'.format(n=num_eig),
                            'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='am',
                                                       num_eig=num_eig, use_full_matrix=use_full_matrix)})
            # methods.append({'name': 'DI_lap{n:0>3d}'.format(n=num_eig),
            #                 'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='lap',
            #                                            num_eig=num_eig, use_full_matrix=use_full_matrix)})
            # methods.append({'name': 'DI_sponge{n:0>3d}'.format(n=num_eig),
            #                 'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='sponge',
            #                                            num_eig=num_eig, use_full_matrix=use_full_matrix)})
    return methods


def run(pid,sim_id):
    config_lists = get_graph_config_lists(sim_id)

    sim = ClassificationSimulation(graph_config_list=config_lists)
    graph_config, percentage_labeled, is_percentage = sim.get_graph_config(pid)
    print('pid:{pid:5d}\t{s} with {p:d}% labels'.format(pid=pid,s=graph_config['model'], p=percentage_labeled))
    method_configs = get_methods(graph_config, sim_id)
    sim.add_method(method_configs)
    sim.run_simulation(pid)
    num_pos = sim.graph.w_pos.count_nonzero()
    num_neg = sim.graph.w_neg.count_nonzero()
    num_tot = sim.graph.weights.count_nonzero()
    # print('ratio in class 1: {r:.2f}'.format(r=np.sum(sim.graph.class_labels)/sim.graph.num_nodes))
    # print(sim.graph.num_nodes, num_tot, num_pos / num_tot, num_neg / num_tot)
    sim.save_results(constants.results_dir['wiki_sim'][sim_id], split_file=False, save_degenerate_stats=False,
                     reduce_data=False)


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
            for i in range(len(constants.results_dir['wiki_sim'])):
                config_lists, name = get_graph_config_lists(sim_id=i, return_name=True)
                sim = ClassificationSimulation(**config_lists)
                print('{n: >3d} configs in simulation {i} --- {name}'.format(n=len(sim.graph_config_list), i=i,
                                                                             name=name))
        elif args[1] == '-p':
            plotting.plot()
        else:
            print('unknown command {c}'.format(c=args[1]))
        pass
    else:
        pid, sim_id = args_to_pid_and_sim_id(sys.argv)
        for p in pid:
            run(p, sim_id)
