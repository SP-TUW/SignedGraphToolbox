from src.launcher.DSLW2022 import constants
from src.launcher import SBMSimulation
from src.node_classification import SpectralLearning
import numpy as np


def make_result_dirs():
    from pathlib import Path
    Path(constants.results_dir['sbm_curve']).mkdir(parents=True, exist_ok=True)


def get_graph_config_lists():
    num_classes_list = [5]
    num_nodes_list = [1500]*3
    class_distribution_list = [[1]*nc for nc in num_classes_list]
    eps_list = np.linspace(0, 0.5, 11)
    percentage_labeled_list = [10]
    config_lists = {'eps_list': eps_list,
                    'percentage_labeled_list': percentage_labeled_list,
                    'sbm_config_list': [{'num_classes': cnd[0], 'num_nodes': cnd[1], 'class_distribution': cnd[2]}
                                          for cnd in zip(num_classes_list,num_nodes_list,class_distribution_list)]}
    return config_lists


def get_methods():
    methods = []
    method = SpectralLearning(num_classes=None)
    methods = [{'method': method, 'name': 'SCLC'}]
    return methods


def run(pid):
    config_lists = get_graph_config_lists()
    method_configs = get_methods()

    sim = SBMSimulation(**config_lists)
    sim.add_method(method_configs)
    sim.run_simulation(pid)
    sim.save_results(constants.results_dir['sbm_curve'], split_file=False)


if __name__ == '__main__':
    import sys
    from src.tools.simulation_tools import args_to_pid

    pid = args_to_pid(sys.argv, get_sim_name=False)
    run(pid)
