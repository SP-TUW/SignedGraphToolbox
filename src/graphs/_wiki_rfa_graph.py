import scipy.io

from ._graph import Graph
from src.tools.graph_tools import get_connected_components
import numpy as np
from scipy.sparse import csr_matrix
import os
import json
import pandas as pd


class WikiRfAGraph(Graph):
    DATA_DIR = os.path.join('data', 'wiki_rfa')

    def __init__(self, do_safe_voter_array=True, combination_method='mean_sign', from_matlab=True, **kwargs):
        if from_matlab:
            w_pos, w_neg, class_labels = self.__load_matlab()
            if combination_method == 'mean_sign':
                weights = w_pos-w_neg
            else:
                weights = w_pos
                w_neg.data[:] = 0
        else:
            voter_array = self.__load_votes(do_safe_voter_array)
            class_labels, weights, w_pos, w_neg = self.__votes_to_l0W(voter_array, combination_method)

        super().__init__(weights=weights, weights_pos=w_pos, weights_neg=w_neg, class_labels=class_labels, num_classes=2, **kwargs)

    def __load_matlab(self):
        mat_wpos = np.genfromtxt(os.path.join(self.DATA_DIR, 'wpos.csv'), delimiter=',')
        mat_wneg = np.genfromtxt(os.path.join(self.DATA_DIR, 'wneg.csv'), delimiter=',')
        mat_l0 = (np.genfromtxt(os.path.join(self.DATA_DIR, 'l0.csv'), delimiter=',',dtype=int) + 1) // 2
        return mat_wpos, mat_wneg, mat_l0

    @staticmethod
    def __votes_to_l0W(voter_df, combination_method='mean_sign'):

        is_self_vote = voter_df['SRC'] == voter_df['TGT']
        voter_df = voter_df[np.bitwise_not(is_self_vote)]

        # find tgt nodes
        tgt_nodes = np.unique(voter_df['TGT'])
        total_nodes = np.unique(np.r_[voter_df['SRC'], voter_df['TGT']])
        num_nodes = tgt_nodes.size
        num_nodes_total = np.max(total_nodes) + 1
        total_to_tgt_map = np.zeros(num_nodes_total,dtype='int')
        total_to_tgt_map[tgt_nodes] = np.arange(num_nodes)

        src = voter_df['SRC']
        tgt = voter_df['TGT']
        dat = voter_df['DAT']
        vot = voter_df['VOT']
        res = voter_df['RES']

        # find set of nodes with class information
        last_vote_df = voter_df.sort_index(ascending=False).groupby('TGT').head(1)
        # last_vote_df = voter_df.sort_values('DAT', ascending=False).groupby('TGT').head(1)
        target = total_to_tgt_map[np.array(last_vote_df['TGT'])]
        l0 = np.zeros(num_nodes, dtype=int)
        l0[target] = (np.array(last_vote_df['RES']) + 1)//2
        # i_sort = np.lexsort(voter_df['TGT'])
        # res_sorted = res[i_sort]
        # tgt_sorted = tgt[i_sort]
        # ind = np.r_[np.flatnonzero(np.diff(tgt_sorted)), tgt.size - 1]
        #
        # l0 = np.zeros(num_nodes, dtype=int)
        # for i, (start,stop) in enumerate(zip(np.r_[0,ind[:-1]+1], ind)):
        #     # map results -1 and +1 to classes 0 and 1
        #     result = (res_sorted[stop] + 1) // 2
        #     target = total_to_tgt_map[tgt_sorted[stop]]
        #     l0[target] = result

        # reduce to relevant nodes only
        keep_votes = voter_df['SRC'].isin(tgt_nodes)
        voter_df = voter_df[keep_votes]

        # generate weight matrix
        w_pos = None
        w_neg = None
        if combination_method == 'mean_sign':
            w_pos = WikiRfAGraph.__get_w_sign(voter_df=voter_df, sign=1, num_nodes=num_nodes, total_to_tgt_map=total_to_tgt_map)
            w_pos = (w_pos+w_pos.T).sign()
            w_neg = WikiRfAGraph.__get_w_sign(voter_df=voter_df, sign=-1, num_nodes=num_nodes, total_to_tgt_map=total_to_tgt_map)
            w_neg = (w_neg+w_neg.T).sign()

            # l0_matlab = np.genfromtxt(os.path.join('data', 'SNAP', 'wiki_rfa', 'l0_full.csv'), delimiter=',')
            # names = []
            # with open(os.path.join('data', 'SNAP', 'wiki_rfa', 'names.txt')) as name_file:
            #     for line in name_file:
            #         names.append(line)
            #
            # i_diff = tgt_nodes[np.flatnonzero((l0_matlab[tgt_nodes]+1)//2-l0)]
            # print('diff at', i_diff)
            # for i in i_diff:
            #     print(names[i])

            w = (w_pos+w_neg).sign()
            w.eliminate_zeros()
        elif combination_method == 'only_pos':
            w_pos = WikiRfAGraph.__get_w_sign(voter_df=voter_df, sign=1, num_nodes=num_nodes, total_to_tgt_map=total_to_tgt_map)
            w_pos = (w_pos+w_pos.T).sign()

            # l0_matlab = np.genfromtxt(os.path.join('data', 'SNAP', 'wiki_rfa', 'l0_full.csv'), delimiter=',')
            # names = []
            # with open(os.path.join('data', 'SNAP', 'wiki_rfa', 'names.txt')) as name_file:
            #     for line in name_file:
            #         names.append(line)
            #
            # i_diff = tgt_nodes[np.flatnonzero((l0_matlab[tgt_nodes]+1)//2-l0)]
            # print('diff at', i_diff)
            # for i in i_diff:
            #     print(names[i])

            w = w_pos.sign()
            w.eliminate_zeros()
        elif combination_method == 'only_neg':
            w_neg = WikiRfAGraph.__get_w_sign(voter_df=voter_df, sign=-1, num_nodes=num_nodes, total_to_tgt_map=total_to_tgt_map)
            w_neg = (w_neg+w_neg.T).sign()

            # l0_matlab = np.genfromtxt(os.path.join('data', 'SNAP', 'wiki_rfa', 'l0_full.csv'), delimiter=',')
            # names = []
            # with open(os.path.join('data', 'SNAP', 'wiki_rfa', 'names.txt')) as name_file:
            #     for line in name_file:
            #         names.append(line)
            #
            # i_diff = tgt_nodes[np.flatnonzero((l0_matlab[tgt_nodes]+1)//2-l0)]
            # print('diff at', i_diff)
            # for i in i_diff:
            #     print(names[i])

            w = w_neg.sign()
            w.eliminate_zeros()
        elif combination_method == 'last_vote':


            # first add reverse voter_df
            votes_reversed = voter_df.copy()
            votes_reversed['SRC'] = voter_df['TGT']
            votes_reversed['TGT'] = voter_df['SRC']
            voter_df = pd.concat((voter_df, votes_reversed), axis=0)

            src = voter_df['SRC']
            tgt = voter_df['TGT']
            dat = voter_df['DAT']
            vot = voter_df['VOT']
            res = voter_df['RES']

            # use only the last vote in chronological order
            sorted_voter_df = voter_df
            last_vote_df = voter_df.sort_values('DAT', ascending=False).groupby(['SRC','TGT']).head(1)
            # i_sort = np.lexsort(voter_df[:, [4, 3, 2, 0, 1]].T)
            # vot_sorted = vot[i_sort]
            # tgt_sorted = tgt[i_sort]
            # src_sorted = src[i_sort]
            #
            # voter_df = voter_df.sort_values(by=['RES','VOT','DAT','SRC','TGT'])
            # tgt_diff = np.flatnonzero(np.diff(tgt_sorted))
            # src_diff = np.flatnonzero(np.diff(src_sorted))
            # ind = np.unique(np.r_[0, tgt_diff, src_diff])

            i = total_to_tgt_map[last_vote_df['SRC']]
            j = total_to_tgt_map[last_vote_df['TGT']]
            v = last_vote_df['VOT']

            w = csr_matrix((v, (i, j)), shape=(num_nodes, num_nodes))
            # w = (w + w.T).sign()
        elif combination_method == 'first_entry':
            # first add reverse voter_df
            votes_reversed = voter_df.copy()
            votes_reversed['SRC'] = voter_df['TGT']
            votes_reversed['TGT'] = voter_df['SRC']
            voter_df = pd.concat((voter_df, votes_reversed), axis=0)

            # use only the last vote in chronological order
            last_vote_df = voter_df.sort_index(ascending=True).reset_index().groupby(['SRC','TGT']).head(1)

            i = total_to_tgt_map[last_vote_df['SRC']]
            j = total_to_tgt_map[last_vote_df['TGT']]
            v = last_vote_df['VOT']

            w = csr_matrix((v, (i, j)), shape=(num_nodes, num_nodes))
        elif combination_method == 'last_entry':
            # first add reverse voter_df
            votes_reversed = voter_df.copy()
            votes_reversed['SRC'] = voter_df['TGT']
            votes_reversed['TGT'] = voter_df['SRC']
            voter_df = pd.concat((voter_df, votes_reversed), axis=0)

            # use only the last vote in chronological order
            last_vote_df = voter_df.sort_index(ascending=False).reset_index().groupby(['SRC','TGT']).head(1)

            i = total_to_tgt_map[last_vote_df['SRC']]
            j = total_to_tgt_map[last_vote_df['TGT']]
            v = last_vote_df['VOT']

            w = csr_matrix((v, (i, j)), shape=(num_nodes, num_nodes))
            w_pos = w.maximum(0)
            w_neg = -w.minimum(0)
        elif combination_method == 'both':
            # first add reverse voter_df
            w_pos = WikiRfAGraph.__get_w_sign(voter_df=voter_df, sign=1, num_nodes=num_nodes, total_to_tgt_map=total_to_tgt_map)
            w_pos = (w_pos+w_pos.T).sign()
            w_neg = WikiRfAGraph.__get_w_sign(voter_df=voter_df, sign=-1, num_nodes=num_nodes, total_to_tgt_map=total_to_tgt_map)
            w_neg = (w_neg+w_neg.T).sign()

            w = (w_pos - w_neg).sign()
        else:
            raise ValueError("unknown combination method ""{m}""".format(m=combination_method))

        ccs = get_connected_components(w)

        w = w[ccs[0], :]
        w = w[:, ccs[0]]
        l0 = l0[ccs[0]]

        w_pos = w_pos[ccs[0], :]
        w_pos = w_pos[:, ccs[0]]

        w_neg = w_neg[ccs[0], :]
        w_neg = w_neg[:, ccs[0]]

        return l0, w, w_pos, -w_neg

    @staticmethod
    def __get_w_sign(voter_df, sign, num_nodes, total_to_tgt_map):
        # returns the accumulated voter_df of one sign (this is directed)
        has_sign = sign * voter_df['VOT'] > 0
        i = total_to_tgt_map[voter_df[has_sign]['SRC']]
        j = total_to_tgt_map[voter_df[has_sign]['TGT']]
        v = voter_df[has_sign]['VOT']
        w = csr_matrix((v, (i, j)), shape=(num_nodes, num_nodes))
        return w

    def __load_votes(self, safe_voter_array):
        data_dir = self.DATA_DIR
        data_loaded = False
        if os.path.isfile(os.path.join(data_dir, 'voter_df.json')):
            try:
                voter_df = pd.read_csv(os.path.join(data_dir, 'voter_df.csv'))
                # with open(os.path.join(data_dir, 'voter_df.json')) as voter_array_file:
                #     voter_df = np.array(json.load(voter_array_file))
                data_loaded = True
            except json.decoder.JSONDecodeError:
                print('voter_df.pkl is corrupted. Generating from scratch.')
        if not data_loaded:
            print("loading from txt file")
            voter_df = self._load_from_txt_file()
            if safe_voter_array:
                voter_df.to_pickle(os.path.join(data_dir, 'voter_df.pkl'))
                voter_df.to_json(os.path.join(data_dir, 'voter_df.json'))
                voter_df.to_csv(os.path.join(data_dir, 'voter_df.csv'))
                # with open(os.path.join(data_dir, 'voter_df.json'), 'w') as voter_array_file:
                #     json.dump(voter_df.tolist(), voter_array_file)
                # np.savetxt(os.path.join(data_dir, 'voter_df.csv'), voter_df, delimiter=",")

        return voter_df

    @staticmethod
    def __is_valid_voter_pair(vp):
        # return vp['vot'] != 0 and vp['src'] != vp['tgt']  # and vp['dat']!=''
        return vp['src'] != vp['tgt']  # and vp['dat']!=''

    @staticmethod
    def __date_to_int(date_str):
        if date_str != '':

            if date_str.startswith('31:29'):
                date_str = date_str.replace('31:29', '03:31')
            date_str = date_str.replace('Jan ', 'January ')
            date_str = date_str.replace('Janry ', 'January ')
            date_str = date_str.replace('January ', '01 ')
            date_str = date_str.replace('Feb ', 'February ')
            date_str = date_str.replace('February ', '02 ')
            date_str = date_str.replace('Mar ', 'March ')
            date_str = date_str.replace('March ', '03 ')
            date_str = date_str.replace('Apr ', 'April ')
            date_str = date_str.replace('April ', '04 ')
            date_str = date_str.replace('Mya ', 'May ')
            date_str = date_str.replace('May ', '05 ')
            date_str = date_str.replace('Jun ', 'June ')
            date_str = date_str.replace('June ', '06 ')
            date_str = date_str.replace('Jul ', 'July ')
            date_str = date_str.replace('Julu ', 'July ')
            date_str = date_str.replace('July ', '07 ')
            date_str = date_str.replace('Aug ', 'August ')
            date_str = date_str.replace('August ', '08 ')
            date_str = date_str.replace('Sep ', 'September ')
            date_str = date_str.replace('September ', '09 ')
            date_str = date_str.replace('Oct ', 'October ')
            date_str = date_str.replace('October ', '10 ')
            date_str = date_str.replace('Nov ', 'November ')
            date_str = date_str.replace('November ', '11 ')
            date_str = date_str.replace('Dec ', 'December ')
            date_str = date_str.replace('December ', '12 ')
            date_str = date_str.replace(':', ' ')
            date_str = date_str.replace(',', '')
            dat_splits = date_str.split(' ')
            Y = dat_splits[4]
            M = dat_splits[3]
            D = '0' * (2 - len(dat_splits[2])) + dat_splits[2]
            h = dat_splits[0]
            m = dat_splits[1]
            dat_str = '{Y}{M}{D}{h}{m}'.format(Y=Y, M=M, D=D, h=h, m=m)
            return int(dat_str)  # datetime.strptime(dat, '%H:%M, %d %B %Y').date()
        else:
            return -1

    @staticmethod
    def __year_to_int(year_str):
        Y = year_str
        M = "00"
        D = "00"
        h = "00"
        m = "00"
        dat_str = '{Y}{M}{D}{h}{m}'.format(Y=Y, M=M, D=D, h=h, m=m)
        return int(dat_str)  # datetime.strptime(dat, '%H:%M, %d %B %Y').date()

    def _load_from_txt_file(self):
        data_dir = WikiRfAGraph.DATA_DIR
        votes = {}
        num_lines = 0

        print('Reading Votes from original data file')
        with open(os.path.join(data_dir, 'wiki-RfA.txt'), encoding='utf8') as raw_data_file:
            for line in raw_data_file:
                if line != '\n':
                    num_lines += 1
                    key = line[:3]
                    val = line[4:-1]
                    if key == 'DAT':
                        val = WikiRfAGraph.__date_to_int(val)
                    elif key == 'YEA':
                        val = WikiRfAGraph.__year_to_int(val)
                    elif key == 'VOT':
                        val = float(val)
                    elif key == 'RES':
                        val = int(val)
                    votes.setdefault(key, [])
                    votes[key].append(val)

        # find list of all names
        src_names = np.array(votes['SRC'])
        tgt_names = np.array(votes['TGT'])
        all_names = np.unique(np.r_[src_names, tgt_names])

        name_file_name = os.path.join(WikiRfAGraph.DATA_DIR, 'names.txt')
        with open(name_file_name, 'w') as name_file:
            for name in all_names:
                name_file.write('{name}\n'.format(name=name))

        voter_df = pd.DataFrame.from_dict(votes)
        voter_df['SRC'] = np.searchsorted(all_names, src_names)
        voter_df['TGT'] = np.searchsorted(all_names, tgt_names)
        voter_df = voter_df.drop('TXT',axis=1)
        voter_df['DAT'] = voter_df[['DAT', 'YEA']].max(axis=1)  # if YEA > DAT then the data in DAT must be invalid
        voter_df = voter_df.drop('YEA',axis=1)


        # voter_df = np.zeros((len(voter_df['SRC']), 5), dtype='int')
        # voter_df[:, 0] = np.searchsorted(all_names, src_names)
        # voter_df[:, 1] = np.searchsorted(all_names, tgt_names)
        # voter_df[:, 2] = np.array(voter_df['DAT'])
        # voter_df[:, 3] = np.array(voter_df['VOT']).astype('int')
        # voter_df[:, 4] = np.array(voter_df['RES']).astype('int')
        #
        # for i in np.flatnonzero(voter_df[:, 2] == -1):
        #     voter_df[i, 2] = voter_df['YEA'][i]

        return voter_df


if __name__ == '__main__':
    # run this module to generate the json representation of the voter array
    # this improves the runtime of subsequent runs significantly faster
    g = WikiRfAGraph(do_safe_voter_array=True)
