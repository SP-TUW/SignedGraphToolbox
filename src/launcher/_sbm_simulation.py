import json
import os
import numpy as np
import time
import warnings

from sklearn.metrics import adjusted_rand_score, f1_score

from ._classification_simulation import ClassificationSimulation


class SBMSimulation(ClassificationSimulation):

    def __init__(self, eps_list, percentage_labeled_list, sbm_config_list, methods_list=None):
        '''

        :param eps_list: specifies the sign flip probabilities for the simulations
        :param percentage_labeled_list: specifies the amount of labels per class
        :param graph_config_list: list of dictionaries with graph parameters. Keys 'num_classes' and 'num_nodes' are required. Other options: 'class_distribution', 'scale_pi', 'scale_pe'
        :param methods_list: list of triplets (name, method, name of initialization method). This class will then build a order of simulation that satisfies the initialization requirements
        '''
        len_num_classes = len(sbm_config_list)
        len_eps = len(eps_list)
        len_percentage = len(percentage_labeled_list)

        graph_config_list = []
        for i_config in range(len_num_classes*len_eps*len_percentage):
            num_classes = sbm_config_list[i_config % len_num_classes]['num_classes']
            num_nodes = sbm_config_list[i_config % len_num_classes]['num_nodes']
            class_distribution = sbm_config_list[i_config % len_num_classes]['class_distribution']
            if 'scale_pi' in sbm_config_list[i_config % len_num_classes].keys():
                scale_pi = sbm_config_list[i_config % len_num_classes]['scale_pi']
            else:
                scale_pi = 1
            if 'scale_pe' in sbm_config_list[i_config % len_num_classes].keys():
                scale_pe = sbm_config_list[i_config % len_num_classes]['scale_pe']
            else:
                scale_pe = 1
            i_config = i_config // len_num_classes
            eps = eps_list[i_config % len_eps]
            i_config = i_config // len_eps
            percentage_labeled = percentage_labeled_list[i_config % len_percentage]
            p = num_classes * (num_classes + 2) / num_nodes
            pi = p * scale_pi
            pe = p * scale_pe

            pi = min(1,max(0,pi))
            pe = min(1,max(0,pe))

            graph_config_list.append({'model': 'SBM',
                            'type': 'Bernoulli',
                            'class_distribution': class_distribution,
                            'num_nodes': num_nodes,
                            'pi': pi,
                            'pe': pe,
                            'li': 0.5 - eps,
                            'le': 0.5 + eps,
                            'percentage_labeled': percentage_labeled,
                            'is_percentage': True,
                            'name':'SBM with {nn} nodes, {nc} classes, and eps={eps} making use of {p:.0%} '
                                   'labels\nscale_pi={pi}, scale_pe={pe}'.format(nn=num_nodes, nc=num_classes,
                                                                                 eps=eps, p=percentage_labeled / 100,
                                                                                 pi=scale_pi, pe=scale_pe),
                            'result_fields': {
                                'eps': eps,
                                'scale_pi': scale_pi,
                                'scale_pe': scale_pe,
                                'num_classes': num_classes}
                            })

        # self.num_classes_list = [c['num_classes'] for c in graph_config_list]
        # self.num_nodes_list = [c['num_nodes'] for c in graph_config_list]
        # self.class_distribution_list = [c['class_distribution'] if 'class_distribution' in c else [1]*c['num_classes']
        #                                 for c in graph_config_list]
        # self.scale_pi = [c['scale_pi'] if 'scale_pi' in c else 1 * c['num_classes'] for c in graph_config_list]
        # self.scale_pe = [c['scale_pe'] if 'scale_pe' in c else 1 * c['num_classes'] for c in graph_config_list]

        super().__init__(graph_config_list=graph_config_list, methods_list=methods_list)

    #     self.method_dict = {}
    #     if methods_list is not None:
    #         self.add_method(methods_list)
    #
    #     self.x = {}
    #     self.l_est = {}
    #     self.t_run = {}
    #     self.sim_id = -1
    #     self.graph = None
    #     self.labels = None
    #
    #
    # def add_method(self, method):
    #     if type(method) is list:
    #         for mc in method:
    #             self.add_method(mc)
    #     else:
    #         name = method['name']
    #         if name not in self.method_dict:
    #             self.method_dict[name] = (method['method'], method['initialization'])
    #         else:
    #             warnings.warn('method already added in the list. overwriting old config')
    #             self.method_dict[name] = method

    # def get_config(self, sim_id):
    #     len_num_classes = len(self.num_classes_list)
    #     len_eps = len(self.eps_list)
    #     len_percentage = len(self.percentage_labeled_list)
    #
    #     i_config = sim_id
    #     num_classes = self.num_classes_list[i_config % len_num_classes]
    #     num_nodes = self.num_nodes_list[i_config % len_num_classes]
    #     class_distribution = self.class_distribution_list[i_config % len_num_classes]
    #     scale_pi = self.scale_pi[i_config % len_num_classes]
    #     scale_pe = self.scale_pe[i_config % len_num_classes]
    #     i_config = i_config // len_num_classes
    #     eps = self.eps_list[i_config % len_eps]
    #     i_config = i_config // len_eps
    #     percentage_labeled = self.percentage_labeled_list[i_config % len_percentage]
    #     p = num_classes * (num_classes + 2) / num_nodes
    #     if self.scale_pe:
    #         pi = p * scale_pi
    #         pe = p * scale_pe
    #     else:
    #         pi = p * scale_pi
    #         pe = p * scale_pi
    #
    #     pi = min(1,max(0,pi))
    #     pe = min(1,max(0,pe))
    #
    #     graph_config = {'model': 'SBM',
    #                     'type': 'Bernoulli',
    #                     'cluster_distribution': class_distribution,
    #                     'num_nodes': num_nodes,
    #                     'pi': pi,
    #                     'pe': pe,
    #                     'li': 0.5 - eps,
    #                     'le': 0.5 + eps,
    #                     'percentage_labeled': percentage_labeled,
    #                     'is_percentage': True,
    #                     'str': 'SBM with {nn} nodes, {nc} classes, and eps={eps} making use of {p:.0%} '
    #                            'labels\nscale_pi={pi}, scale_pe={pe}'.format(nn=num_nodes, nc=num_classes, eps=eps,
    #                                                                          p=percentage_labeled / 100, pi=scale_pi,
    #                                                                          pe=scale_pe),
    #                     'result_fields': {
    #                         'eps': eps,
    #                         'scale_pi': scale_pi,
    #                         'scale_pe': scale_pe,
    #                         'num_classes': num_classes}
    #                     }
    #     return graph_config, num_nodes, num_classes, eps, percentage_labeled, scale_pi, scale_pe

    # def get_graph(self, sim_id):
    #     graph_config, num_nodes, num_classes, eps, percentage_labeled, scale_pi, scale_pe = self.get_config(sim_id)
    #     graph = StochasticBlockModel(**graph_config)
    #     print('SBM with {nn} nodes, {nc} classes, and eps={eps} making use of {p:.0%} labels'.format(
    #         nn=num_nodes, nc=num_classes, eps=eps, p=percentage_labeled / 100))
    #     print('scale_pi={pi}, scale_pe={pe}'.format(pi=scale_pi, pe=scale_pe))
    #
    #     labels = select_labels(graph, percentage_labeled, is_percentage=True)
    #     return graph, labels

    # def run_simulation(self, sim_id):
    #     np.random.seed(sim_id)
    #     self.graph, self.labels = self.get_graph(sim_id)
    #
    #     for method in self.method_dict:
    #         name = method.pop('name')
    #         print('method: {name}'.format(name=name))
    #         if 'num_clusters' not in method.keys() and 'num_classes' in method.keys():
    #             method['num_clusters'] = method.pop('num_classes')
    #         elif 'num_clusters' not in method.keys():
    #             method['num_clusters'] = self.graph.K0
    #         if 'x0_choice' in method.keys():
    #             if method['x0_choice'] == 'rand':
    #                 l = np.random.randint(0, self.graph.K0, self.graph.N)
    #             else:
    #                 l = self.l_est[method['x0_choice']]
    #             method.pop('x0_choice')
    #             x0 = -np.ones((self.graph.N, self.graph.K0))
    #             x0[range(self.graph.N), l] = 1
    #         else:
    #             x0 = 0
    #         t_start = time.time()
    #         self.x[name], self.l_est[name] = cluster_wrapper.cluster(self.graph, labels=self.labels, x0=x0, **method['config'])
    #         t_stop = time.time()
    #         self.t_run[name] = t_stop-t_start
    #         n_err_sep = np.sum(self.l_est[name] != self.graph.l0)
    #         print('n_err_{name}={n}'.format(name=name, n=n_err_sep))
    #     self.sim_id = sim_id

    # def save_results(self, results_dir, split_file=False):
    #     if self.sim_id >= 0:
    #         graph_config, num_nodes, num_classes, eps, percentage_labeled, scale_pi, scale_pe = self.get_config(self.sim_id)
    #         results = {'pid': self.sim_id,
    #                    'eps': eps,
    #                    'scale_pi': scale_pi,
    #                    'scale_pe': scale_pe,
    #                    'num_classes': num_classes,
    #                    'graph_config': graph_config,
    #                    'percentage_labeled': percentage_labeled,
    #                    'num_nodes': num_nodes}
    #
    #         num_labels = len(self.labels['i'])
    #         for name, l_est in self.l_est.items():
    #             is_wrong = np.array(self.graph.l0) != np.array(l_est)
    #             is_label = np.zeros(num_nodes, dtype=bool)
    #             is_label[self.labels['i']] = True
    #             n_err_total = int(np.sum(is_wrong))
    #             n_err_labeled = int(np.sum(is_wrong[is_label]))
    #             n_err_unlabeled = n_err_total - n_err_labeled
    #             acc_total = 1 - n_err_total / num_nodes
    #             acc_labeled = 1 - n_err_labeled / num_labels
    #             acc_unlabeled = 1 - n_err_unlabeled / (num_nodes - num_labels)
    #             ari = adjusted_rand_score(l_est, self.graph.l0)
    #             f1_micro = f1_score(self.graph.l0, l_est, average='micro')
    #             f1_macro = f1_score(self.graph.l0, l_est, average='macro')
    #
    #             if not split_file:
    #                 results['n_err_total_{n}'.format(n=name)] = n_err_total
    #                 results['n_err_labeled_{n}'.format(n=name)] = n_err_labeled
    #                 results['n_err_unlabeled_{n}'.format(n=name)] = n_err_unlabeled
    #                 results['acc_total_{n}'.format(n=name)] = acc_total
    #                 results['acc_labeled_{n}'.format(n=name)] = acc_labeled
    #                 results['acc_unlabeled_{n}'.format(n=name)] = acc_unlabeled
    #                 results['ari_{n}'.format(n=name)] = ari
    #                 results['f1_micro_{n}'.format(n=name)] = f1_micro
    #                 results['f1_macro_{n}'.format(n=name)] = f1_macro
    #                 results['t_run_{n}'.format(n=name)] = self.t_run[name]
    #             else:
    #                 results[name] = {'n_err_total': n_err_total,
    #                                  'n_err_labeled': n_err_labeled,
    #                                  'n_err_unlabeled': n_err_unlabeled,
    #                                  'acc_total': acc_total,
    #                                  'acc_labeled': acc_labeled,
    #                                  'acc_unlabeled': acc_unlabeled,
    #                                  'ari': ari,
    #                                  'f1_micro': f1_micro,
    #                                  'f1_macro': f1_macro,
    #                                  't_run': self.t_run[name]}
    #
    #         if not split_file:
    #             results_file_name = os.path.join(results_dir, '{id}.json'.format(id=self.sim_id))
    #             with open(results_file_name, 'weights') as results_file:
    #                 json.dump(results, results_file)
    #         else:
    #             for name in self.l_est.keys():
    #                 result = results[name]
    #                 result['name'] = name
    #                 result['pid'] = self.sim_id
    #                 result['eps'] = eps
    #                 result['scale_pi'] = scale_pi
    #                 result['scale_pe'] = scale_pe
    #                 result['graph_config'] = graph_config
    #                 result['percentage_labeled'] = percentage_labeled
    #                 result['num_nodes'] = num_nodes
    #                 result['num_classes'] = num_classes
    #
    #                 results_file_name = os.path.join(results_dir, '{id}_{n}.json'.format(id=self.sim_id, n=name))
    #                 with open(results_file_name, 'weights') as results_file:
    #                     json.dump(result, results_file)
    #
    #     else:
    #         raise RuntimeError('need to run the simulation before saving results')
