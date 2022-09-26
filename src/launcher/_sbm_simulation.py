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


        super().__init__(graph_config_list=graph_config_list, methods_list=methods_list)