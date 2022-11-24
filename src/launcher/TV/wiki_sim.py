from src.launcher.TV import constants
from src.launcher import ClassificationSimulation
from src.launcher.TV import wiki_plotting as plotting
from src.node_classification import DiffuseInterface, SpectralLearning, TvAugmentedADMM


def make_result_dirs():
    pass


def combine_results():
    pass


def get_graph_config_lists(sim_id, return_name=False):
    config_lists = []
    if sim_id == 0:  # wiki editor
        name = 'WIKI_EDITOR'
    elif sim_id == 1:  # wiki elec
        name = 'WIKI_ELEC'
    elif sim_id == 2:  # wiki RfA
        name = 'WIKI_RFA'
    else:
        raise ValueError('unknown sim_id')

    for p in [1, 5, 10, 15]:
        config_lists.append({'model': name, 'percentage_labeled': p, 'is_percentage': True, 'name': '{n}{p:0>2d}'.format(n=name,p=p)})

    if return_name:
        return config_lists, name
    else:
        return config_lists


def get_methods(graph_config, sim_id):
    num_classes = 2
    v = 1
    methods = [
        {'name': 'sncBNC', 'method': SpectralLearning(num_classes=num_classes, objective='BNC_INDEF')},
        # {'name': 'sncSponge', 'method': SpectralLearning(num_classes=num_classes, objective='SPONGE')},
    ]
    # b = 1e4
    # pre = 0
    # l_guess = 'sncSponge'
    # methods.append({'name': 'tv_nc_beta{penalty:0>+1.1f}_pre{pre}_{guess}'.format(penalty=np.log10(b), pre=int(pre),
    #                                                                               guess=l_guess),
    #                 'l_guess': l_guess, 'is_unsupervised': False,
    #                 'method': TvStandardADMM(num_classes=num_classes, verbosity=v, penalty_parameter=b,
    #                                          pre_iteration_version=pre, t_max_no_change=None)})
    methods.append({'name': 'tv30', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v,
                                                              degenerate_heuristic=None,
                                                              eps_rel=10 ** (-30 / 10),
                                                              eps_abs=10 ** (-30 / 10),
                                                              resampling_x_min=90 / 100)})
    methods.append({'name': 'tv15_resampling05', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v,
                                                                           degenerate_heuristic='rangapuram_resampling',
                                                                           eps_rel=10 ** (-15 / 10),
                                                                           eps_abs=10 ** (-15 / 10),
                                                                           resampling_x_min=5 / 100)})
    methods.append({'name': 'tv15_regularize90', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v,
                                                                           degenerate_heuristic='regularize',
                                                                           eps_rel=10 ** (-15 / 10),
                                                                           eps_abs=10 ** (-15 / 10),
                                                                           resampling_x_min=90 / 100)})
    methods.append({'name': 'tv30_regularize90', 'method': TvAugmentedADMM(num_classes=num_classes, verbosity=v,
                                                                           degenerate_heuristic='regularize',
                                                                           eps_rel=10 ** (-30 / 10),
                                                                           eps_abs=10 ** (-30 / 10),
                                                                           resampling_x_min=90 / 100)})

    if graph_config['model'] in ['WIKI_ELEC', 'WIKI_RFA']:
        num_eig = 20
    else:
        num_eig = 100

    methods.append({'name': 'diffuseInterface_sym{n:0>3d}'.format(n=num_eig),
                    'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='sym',
                                               num_eig=num_eig, use_full_matrix=True)})
    methods.append({'name': 'diffuseInterface_am{n:0>3d}'.format(n=num_eig),
                    'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='am',
                                               num_eig=num_eig, use_full_matrix=True)})
    methods.append({'name': 'diffuseInterface_lap{n:0>3d}'.format(n=num_eig),
                    'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='lap',
                                               num_eig=num_eig, use_full_matrix=True)})
    methods.append({'name': 'diffuseInterface_sponge{n:0>3d}'.format(n=num_eig),
                    'method': DiffuseInterface(num_classes=num_classes, verbosity=v, objective='sponge',
                                               num_eig=num_eig, use_full_matrix=True)})
    return methods


def run(pid,sim_id):
    config_lists = get_graph_config_lists(sim_id)

    sim = ClassificationSimulation(graph_config_list=config_lists)
    graph_config, percentage_labeled, is_percentage = sim.get_graph_config(pid)
    print('{s} with {p:d}% labels'.format(s=graph_config['model'], p=percentage_labeled))
    method_configs = get_methods(graph_config, sim_id)
    sim.add_method(method_configs)
    sim.run_simulation(pid)
    sim.save_results(constants.results_dir['sbm_sim'][sim_id], split_file=False, save_degenerate_stats=False,
                     reduce_data=True)


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
        run(pid, sim_id)
