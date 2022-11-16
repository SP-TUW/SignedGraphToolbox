import json
import os
import time
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score

from src.graphs import graph_factory
from src.tools.graph_tools import calc_signed_cut
from src.tools.graph_tools import select_labels
from src.tools.simulation_tools import find_min_err_label_permutation


class ClassificationSimulation:

    def __init__(self, graph_config_list=None, methods_list=None):
        self.graph_config_list = graph_config_list

        self.methods_list = []
        if methods_list is not None:
            self.add_method(methods_list)

        self.embedding = {}
        self.normalized_embedding = {}
        self.l_est = {}
        self.t_run = {}
        self.n_err = {}
        self.cut = {}
        self.sim_id = -1
        self.graph = None
        self.labels = None
        self.current_graph_config = None
        self.current_percentage_labeled = None
        self.current_is_percentage = None

    #
    # def add_method(self, method_config):
    #     if type(method_config) is list:
    #         for mc in method_config:
    #             self.add_method(mc)
    #     else:
    #         name = method_config['name']
    #         if name not in self.method_config_list:
    #             self.method_config_list[name] = (method_config['method'], method_config['initialization'])
    #         else:
    #             warnings.warn('method already added in the list. overwriting old config')
    #             self.method_config_list[name] = method_config


    def add_method(self, method):
        '''

        :param method: a dict with necessary fields 'name' and 'config' and optional field 'initialization'. If
        'initialization' is present then the result of this method will be used as initial guess for the current
        method :return:
        '''
        if type(method) is list:
            for method in method:
                self.add_method(method)
        else:
            # gather names of all already added methods
            methods_names = [method['name'] for method in self.methods_list]
            # assert that current method has a name
            assert 'name' in method.keys()
            # add if name not already in list
            if method['name'] in methods_names:
                warnings.warn('method already added in the list. Ovewriting old config')
            if 'initialization' in method.keys() and method['initialization'] not in methods_names:
                raise ValueError(method['initialization'] + ' not yet added to the list of methods. Please add dependent methods after their dependency')
            self.methods_list.append(method)

    def add_graphs(self, graph_config):
        if type(graph_config) is list:
            for graph in graph_config:
                self.add_graphs(graph)
        else:
            # gather names of all already added graph configs
            graph_names = [graph['name'] for graph in self.graph_config_list]
            assert 'name' in graph_config.keys()
            assert 'percentage_labeled' in graph_config.keys()
            assert 'is_percentage' in graph_config.keys()
            # add if name not already in list
            if graph_config['name'] not in graph_names:
                self.graph_config_list.append(graph_config)
            else:
                warnings.warn('graph config already added in the list')

    def get_graph_config(self, sim_id):
        graph_config = self.graph_config_list[sim_id % len(self.graph_config_list)].copy()
        percentage_labeled = graph_config.pop('percentage_labeled')
        is_percentage = graph_config.pop('is_percentage')
        return graph_config, percentage_labeled, is_percentage

    def __get_graph(self, sim_id):
        graph_config, percentage_labeled, is_percentage = self.get_graph_config(sim_id)
        graph = graph_factory.make_graph(**graph_config)
        if 'str' in graph_config.keys():
            print(graph_config['str'])

        labels = select_labels(graph, percentage_labeled, is_percentage=is_percentage)
        return graph, labels

    def run_simulation(self, sim_id):
        np.random.seed(sim_id)
        graph_config, percentage_labeled, is_percentage = self.get_graph_config(sim_id)
        self.graph, self.labels = self.__get_graph(sim_id)
        print('running simulations for {n}'.format(n=graph_config['name']))

        print('cut gt={c}'.format(c=calc_signed_cut(self.graph.weights, self.graph.class_labels)))


        for method in self.methods_list:
            name = method.pop('name')
            print('method: {name}'.format(name=name))
            # if 'num_classes' not in method.keys() and 'num_classes' in method.keys():
            #     method['num_classes'] = method.pop('num_classes')
            # elif 'num_classes' not in method.keys():
            #     method['num_classes'] = self.graph.K0

            if 'l_guess' in method and method['l_guess'] not in ['min_err','min_cut']:
                l_guess = self.l_est[method['l_guess']]
            elif 'l_guess' in method and method['l_guess'] == 'min_err':
                guess = min(self.n_err,key=self.n_err.get)
            else:
                l_guess = None
            t_start = time.time()
            if 'is_unsupervised' in method and method['is_unsupervised']:
                l_est = method['method'].estimate_labels(self.graph, labels=None, guess=l_guess)
            else:
                l_est = method['method'].estimate_labels(self.graph, labels=self.labels, guess=l_guess)
            t_stop = time.time()
            if ('is_unsupervised' in method and method['is_unsupervised']) or self.labels is None or len(self.labels['i']) == 0:
                self.l_est[name] = find_min_err_label_permutation(l_est, self.graph.class_labels, self.graph.num_classes, self.graph.num_classes)
            else:
                self.l_est[name] = l_est
            self.embedding[name] = method['method'].embedding
            self.normalized_embedding[name] = method['method'].normalized_embedding
            self.t_run[name] = t_stop - t_start
            self.n_err[name] = np.sum(self.l_est[name] != self.graph.class_labels)
            self.cut[name] = calc_signed_cut(self.graph.weights, self.l_est[name])
            print('n_err_{name}={n}'.format(name=name, n=self.n_err[name]))
            print('cut {name}={c}'.format(name=name, c=self.cut[name]))
        self.current_graph_config = graph_config.copy()
        self.current_percentage_labeled = percentage_labeled
        self.current_is_percentage = is_percentage
        self.sim_id = sim_id




    def save_results(self, results_dir, split_file=False, save_degenerate_stats=False):
        if self.sim_id >= 0:
            # graph_config, percentage_labeled, is_percentage = self.__get_config(self.sim_id)
            # graph_config, num_nodes, num_classes, eps, percentage_labeled, scale_pi, scale_pe = self.get_config(self.sim_id)
            num_nodes = self.graph.num_nodes
            num_classes = self.graph.num_classes
            num_labels = len(self.labels['i'])

            # embedding_gt = -np.ones((num_nodes,num_classes))
            # embedding_gt[np.arange(num_nodes), self.graph.class_labels] = 1
            cut_gt = calc_signed_cut(self.graph.weights,self.graph.class_labels)

            results = {}
            results__ = {'pid': self.sim_id,
                         'graph_name': self.current_graph_config['name'],
                         'graph_config': self.current_graph_config,
                         'percentage_labeled': self.current_percentage_labeled,
                         'num_nodes': num_nodes,
                         'cut_gt': cut_gt
                         }

            if 'result_fields' in self.current_graph_config.keys():
                results__.update(self.current_graph_config['result_fields'])

            if not split_file:
                results = results__

            for name, l_est in self.l_est.items():
                is_wrong = np.array(self.graph.class_labels) != np.array(l_est)
                is_label = np.zeros(num_nodes, dtype=bool)
                is_label[self.labels['i']] = True
                n_err_total = int(np.sum(is_wrong))
                n_err_labeled = int(np.sum(is_wrong[is_label]))
                n_err_unlabeled = n_err_total - n_err_labeled
                acc_total = 1 - n_err_total / num_nodes
                acc_labeled = 1 - n_err_labeled / max(1,num_labels)
                acc_unlabeled = 1 - n_err_unlabeled / (num_nodes - num_labels)
                ari = adjusted_rand_score(l_est, self.graph.class_labels)
                f1_micro = f1_score(self.graph.class_labels, l_est, average='micro')
                f1_macro = f1_score(self.graph.class_labels, l_est, average='macro')

                # embedding = -np.ones((num_nodes,num_classes))
                # embedding[np.arange(num_nodes), l_est] = 1
                cut = calc_signed_cut(self.graph.weights,l_est)

                results_ = {'n_err_total': n_err_total,
                            'n_err_labeled': n_err_labeled,
                            'n_err_unlabeled': n_err_unlabeled,
                            'acc_total': acc_total,
                            'acc_labeled': acc_labeled,
                            'acc_unlabeled': acc_unlabeled,
                            'ari': ari,
                            'f1_micro': f1_micro,
                            'f1_macro': f1_macro,
                            'cut': cut,
                            't_run': self.t_run[name]}

                if save_degenerate_stats:
                    for i in range(8,20):
                        results_['num_degenerate{i}'.format(i=int(5*i))] = int(np.sum(np.max(self.normalized_embedding[name], axis=1) <= i/20))

                if not split_file:
                    keys = ['{k}_{n}'.format(k=key, n=name) for key in results_.keys()]
                    results_ = dict(zip(keys,list(results_.values())))
                    results.update(results_)
                else:
                    results[name] = results_
                    results[name]['method_name'] = name
                    results[name].update(results__)

            if not split_file:
                results_file_name = os.path.join(results_dir, '{id}.json'.format(id=self.sim_id))
                with open(results_file_name, 'w') as results_file:
                    json.dump(results, results_file)
            else:
                for name in self.l_est.keys():
                    result = results[name]
                    results_file_name = os.path.join(results_dir, '{id}_{n}.json'.format(id=self.sim_id, n=name))
                    with open(results_file_name, 'w') as results_file:
                        json.dump(result, results_file)

        else:
            raise RuntimeError('need to run the simulation before saving results')
