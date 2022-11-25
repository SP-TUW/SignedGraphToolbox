import pandas as pd
import scipy.io

from src.graphs._wiki_rfa_graph import WikiRfAGraph
import numpy as np
import os
import codecs
from enum import Enum


class WikiElecGraph(WikiRfAGraph):
    DATA_DIR = os.path.join('data', 'wiki_elec')

    def _load_from_txt_file(self):
        filename = os.path.join(WikiElecGraph.DATA_DIR, 'wikiElec.ElecBs3.txt')
        # states
        # E: get E
        # T: get T
        # U: get U
        # N: get N
        # V: get all voter_df
        states = Enum('states', 'E T U N V')
        state = states.E
        elections = []
        num_votes = 0
        names = np.array([''], dtype=object)

        def insert_name(id, name, names):
            if id >= names.size:
                len_diff = id + 1 - names.size
                names = np.r_[names, np.array([''] * len_diff, dtype=object)]
            names[id] = name
            return names

        with codecs.open(filename, 'r', encoding='latin-1') as file:
            election = {}
            # try:
            for line in file:
                # print(line)
                line_split = line.split('\t')
                if state == states.E and line_split[0] != 'E':
                    election = {}
                elif state == states.E and line_split[0] == 'E':
                    election['E'] = int(line_split[1])
                    state = states.T
                elif state == states.T and line_split[0] == 'T':
                    election['T'] = line_split[1]
                    state = states.U
                elif state == states.U and line_split[0] == 'U':
                    id = int(line_split[1])
                    name = line_split[2][:-2]
                    election['U'] = id
                    names = insert_name(id, name, names)
                    state = states.N
                elif state == states.N and line_split[0] == 'N':
                    id = int(line_split[1])
                    name = line_split[2][:-2]
                    election['N'] = id
                    names = insert_name(id, name, names)
                    state = states.V
                elif state == states.V and line_split[0] == 'V':
                    vot = float(line_split[1])
                    id = int(line_split[2])
                    dat_str = line_split[3].translate({ord(c): None for c in '- :'})
                    dat = int(dat_str)
                    name = line_split[4][:-2]
                    election.setdefault('V',[[], [], []])
                    election['V'][0].append(id)
                    election['V'][1].append(vot)
                    election['V'][2].append(dat)
                    num_votes += 1
                    names = insert_name(id, name, names)
                elif state == states.V and line_split[0] != '':
                    elections.append(election)
                    election = {}
                    state = states.E
                else:
                    raise ValueError('something is wrong in this line\n{line}'.format(line=line))
            # except UnicodeDecodeError:
            #     a = 1
            #     pass
        id_to_index = np.zeros(names.size,dtype=int)
        num_names = np.sum(names != '')
        i_sort = np.argsort(names)[-num_names:]
        id_to_index[i_sort] = np.arange(num_names)
        voter_array = np.zeros((num_votes, 5), dtype=int)
        i_vote = 0
        for election in elections:
            election_size = len(election['V'][0])
            # print(election['U'])
            voter_array[i_vote:i_vote + election_size, 0] = id_to_index[election['V'][0]]
            voter_array[i_vote:i_vote + election_size, 1] = id_to_index[election['U']]
            voter_array[i_vote:i_vote + election_size, 2] = election['V'][2]
            voter_array[i_vote:i_vote + election_size, 3] = election['V'][1]
            voter_array[i_vote:i_vote + election_size, 4] = election['E']
            i_vote += election_size

        voter_df = pd.DataFrame(voter_array,columns=['SRC','TGT','DAT','VOT','RES'])
        voter_df['VOT'] = voter_df['VOT'].astype(float)

        return voter_df


if __name__ == '__main__':
    # run this module to generate the json representation of the voter array
    # this improves the runtime of subsequent runs significantly faster
    g = WikiElecGraph(do_safe_voter_array=True)
