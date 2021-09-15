# This module builds on the code provided here: 
# https://dmm.anu.edu.au/lsdbook2020/lsd_eval_programs-20201030.zip \   
        
# framework.py - Module that contains core classes for the evaluation framework
#
# Ali Arous
#
# Contact: eng.aliarous@gmail.com
#
# Technische Universität Berlin 2021
# -----------------------------------------------------------------------------
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =============================================================================
           
from typing import List, Tuple, Iterable, Set, Optional, Any, Dict
from abc import ABC, abstractmethod
from bitarray import bitarray
from reporting import *
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison
from linkageUtils import *
from linkageUtils import *
import io
import gzip, hashlib, itertools, math, os, random, sys, time, binascii, numpy, pandas as pd
import matplotlib.pyplot as plt
import encoding  # Bloom filter encoding module
import hashing  # Bloom filter hashing module
import blocking
#



class HashingMethod(ABC):
    def set_params(self, bf_len, num_hash_funct):
        self.bf_len = bf_len
        self.num_hash_funct = num_hash_funct

    @abstractmethod
    def hash_q_gram_set(self, q_gram_set: Set[str]) -> bitarray:
        pass


class EncodingMethod(ABC):
    def __init__(self, type: str):
        """Initialise the Encoding Method by providing the required parameters.
        params:
            attr_encode_tuple_list: list[ (attr_num, q, padded, hash_method, num_bf_bit) ],
            type: string description of the encoding method type (e.g., 'CLK')
        """
        self.type = type

    def set_encoding_params(self, attr_encode_tuple_list: Iterable[Any]):
        self.attr_encode_tuple_list = attr_encode_tuple_list

    @abstractmethod
    def encode(self, attr_val_list: List[str]) -> bitarray:
        """Encode values in the given 'attr_val_list' according to the settings
           provided in the 'attr_encode_tuple_list'."""
        pass


class PreEncodingBlockingMethod(ABC):
    def __init__(self):
        self.type="pre_encoding"

    @abstractmethod
    def generate_blocks(self, rec_q_gram_dict: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        params:
            rec_q_gram_dict:  The dictionary of q-gram sets, in the form: dict[ record_id, set of q-gram tokens ]
        returns:
            dict [block_id, set of record_id]
        """
        pass

class PostEncodingBlockingMethod(ABC):
    def __init__(self):
        self.type="post_encoding"

    @abstractmethod
    def generate_blocks(self, rec_bf_dict: Dict[str, Set[bitarray]]) -> Dict[str, Set[str]]:
        """
        params:
            rec_q_gram_dict:  The dictionary of q-gram sets, in the form: dict[ record_id, set of Bloom filters ]
        returns:
            dict [block_id, set of record_id]
        """
        pass


class Pipeline:
    def __init__(self, encoding_method=None, blocking_method=None):
        self.encode_method = encoding_method
        self.block_method = blocking_method


    def run_encoding(self, rec_attr_val_dict1, header_list1, rec_attr_val_dict2, header_list2):
        if self.encode_method.type == 'clk-rbf':
            self.rec_bf_dict1 = gen_bf_dict_external(rec_attr_val_dict1, self.encode_method)
            self.rec_bf_dict2 = gen_bf_dict_external(rec_attr_val_dict2, self.encode_method)
        else:
            self.rec_bf_dict1 = gen_bf_dict(rec_attr_val_dict1, self.encode_method)
            self.rec_bf_dict2 = gen_bf_dict(rec_attr_val_dict2, self.encode_method)


    def run_blocking(self, rec_q_gram_dict1=None, rec_q_gram_dict2=None):
        if self.block_method == None:
            self.block_dict1 = {'all': set(rec_q_gram_dict1.keys())}
            self.block_dict2 = {'all': set(rec_q_gram_dict2.keys())}

        elif self.block_method.type == 'pre_encoding':
            self.block_dict1 = self.block_method.generate_blocks(rec_q_gram_dict1)
            self.block_dict2 = self.block_method.generate_blocks(rec_q_gram_dict2)
        else:
            self.block_dict1 = self.block_method.generate_blocks(self.rec_bf_dict1)
            self.block_dict2 = self.block_method.generate_blocks(self.rec_bf_dict2)

    def get_blocking_statistics(self, true_match_set):
        tot_num_true_matches = len(true_match_set)
        match_rec_id_set = set()
        cand_rec_num = 0
        num_matches = 0
        for key, val_set in self.block_dict1.items():
            if key in self.block_dict2:
                match_rec_id_set.update(val_set.intersection(self.block_dict2[key]))
                cand_rec_num+=len(val_set)*len(self.block_dict2[key])

        num_matches = len(match_rec_id_set)

        # pairs-completeness
        pc = float(num_matches) / float(tot_num_true_matches)
        return {'pc': pc, 'num_matches': num_matches, 'cand_rec_num': cand_rec_num}


    def conduct_linkage(self, min_sim):
        assert min_sim >= 0.0 and min_sim <= 1.0, min_sim
        rec_pair_dict = {}
        # Keep track of pairs compared so each pair is only compared once
        #
        pairs_compared_set = set()

        # Iterate over all block values that occur in both data sets
        #
        for block_val1 in self.block_dict1.keys():

            if (block_val1 not in self.block_dict2):
                continue  # Block value not in common, go to next one

            ent_id_set1 = self.block_dict1[block_val1]
            ent_id_set2 = self.block_dict2[block_val1]

            # Iterate over all value pairs
            #
            for ent_id1 in ent_id_set1:
                val1 = self.rec_bf_dict1[ent_id1]

                for ent_id2 in ent_id_set2:

                    ent_id_pair = (ent_id1, ent_id2)

                    if (ent_id_pair not in pairs_compared_set):

                        val2 = self.rec_bf_dict2[ent_id2]

                        pairs_compared_set.add(ent_id_pair)

                        sim = cal_bf_sim(val1, val2)  # Calculate the similarity

                        if (sim >= min_sim):
                            rec_pair_dict[ent_id_pair] = sim

        num_all_comparisons = len(self.rec_bf_dict1) * len(self.rec_bf_dict2)
        num_pairs_compared = len(pairs_compared_set)
        return rec_pair_dict


class Report:
    def __init__(self):
        self.reset_linkage_quality_containers()
        self.time_dict = {}
        self.meas = []
        self.blocking_statistics = None

    def reset_linkage_quality_containers(self):
        self.q_gram_prec_list = []
        self.q_gram_reca_list = []
        self.q_gram_fmes_list = []
        self.enc_prec_list = []
        self.enc_reca_list = []
        self.enc_fmes_list = []

        self.q_gram_tp_list = []
        self.q_gram_fp_list = []
        self.q_gram_fn_list = []
        self.q_gram_tn_list = []
        self.enc_tp_list = []
        self.enc_fp_list = []
        self.enc_fn_list = []
        self.enc_tn_list = []

        self.time_dict = {}


    def add_time(self, k, v):
        self.time_dict[k] = v

    def append_meas(self, mea):
        self.meas = self.meas + mea

    def init_blocking_report(self, encode_type_list):
        meas_labels = [
            'Blocking time (sec)',
            'Linkage time (sec)',
            'Number of matches found',
            'Number of candidate pairs',
            'Pairs Completeness',
            'Reduction Ratio'
        ]
        self.tab_blocking = {
            'multi_index': [meas_labels, encode_type_list],  # list of lists
            'multi_index_names': ['Measures', 'Method'],
            'cols': {

            }
        }

    def init_report(self, sim_thres_list, q_gram_class_res_dict):
        for sim_thres in sim_thres_list:
            (tp, fp, tn, fn) = q_gram_class_res_dict[sim_thres]

            self.q_gram_tp_list.append(tp)
            self.q_gram_fp_list.append(fp)
            self.q_gram_fn_list.append(fn)
            self.q_gram_tn_list.append(tn)

            prec = calc_precision(tp, fp, fn, tn)
            self.q_gram_prec_list.append(prec)
            reca = calc_recall(tp, fp, fn, tn)
            self.q_gram_reca_list.append(reca)
            fmes = calc_fmeasure(tp, fp, fn, tn)
            self.q_gram_fmes_list.append(fmes)

    def update_report(self, sim_thres_list, bf_class_res_dict):
        bf_prec_list = []
        bf_reca_list = []
        bf_fmes_list = []

        bf_tp_list = []
        bf_fp_list = []
        bf_fn_list = []
        bf_tn_list = []

        for sim_thres in sim_thres_list:
            (tp, fp, tn, fn) = bf_class_res_dict[sim_thres]

            bf_tp_list.append(tp)
            bf_fp_list.append(fp)
            bf_fn_list.append(fn)
            bf_tn_list.append(tn)

            prec = calc_precision(tp, fp, fn, tn)
            bf_prec_list.append(prec)
            reca = calc_recall(tp, fp, fn, tn)
            bf_reca_list.append(reca)
            fmes = calc_fmeasure(tp, fp, fn, tn)
            bf_fmes_list.append(fmes)

        self.enc_prec_list.append(bf_prec_list)
        self.enc_reca_list.append(bf_reca_list)
        self.enc_fmes_list.append(bf_fmes_list)

        self. enc_tp_list.append(bf_tp_list)
        self.enc_fp_list.append(bf_fp_list)
        self.enc_fn_list.append(bf_fn_list)
        self.enc_tn_list.append(bf_tn_list)

    def report_linkage_time(self):
        print()
        print('Time used for q-gram linkage: %.2f sec' % (self.time_dict['q-gram']))
        self.time_dict.pop('q-gram')
        for m_k in self.time_dict:
            print(m_k.rjust(len('Time used for q-gram')), end='')
            print(' linkage: %.2f sec' % (self.time_dict[m_k]))
        print()

    def report_blocking_meas(self, num_rec):
        # self.tab_blocking = tab_blocking

        self.re_ordered_meas = []
        times = len(self.tab_blocking['multi_index'][1])
        shift = len(self.tab_blocking['multi_index'][0])

        for i in range(len(self.tab_blocking['multi_index'][0])):
            acc = 0
            for j in range(times):
                self.re_ordered_meas += [self.meas[i + acc]]
                acc += shift

        self.tab_blocking['cols'][num_rec] = self.re_ordered_meas
        display_blocking(self.tab_blocking)

    def report_linkage_quality_meas(self, encode_type_list, sim_thres_list, conf_str):
        # Linkage quality measures
        tab = {
            'multi_index': [['q-gram'] + encode_type_list, sim_thres_list],  # list of lists
            'multi_index_names': ['Method', 'Sim. thres.'],  # list of strings
            'columns': {
                'TP': [self.q_gram_tp_list] + self.enc_tp_list,
                'FP': [self.q_gram_fp_list] + self.enc_fp_list,
                'FN': [self.q_gram_fn_list] + self.enc_fn_list,
                #             'TN': [q_gram_tn_list]+enc_tn_list,
                'Precision': [self.q_gram_prec_list] + self.enc_prec_list,  # list of lists (i.e., list of all precisions)
                'Recall': [self.q_gram_reca_list] + self.enc_reca_list,
                'F-Meas.': [self.q_gram_fmes_list] + self.enc_fmes_list
            }
        }
        display_df(tab, conf_str=conf_str, save_table=self.save_table)

    def plot_linkage_quality_fig(self, sim_thres_list, legend_str_list):
        plot_it(sim_thres_list, self.q_gram_prec_list, self.q_gram_reca_list, self.enc_prec_list, self.enc_reca_list, legend_str_list)


def main():
    data_set_file_name1 = 'Data/totest/febrl4_A_sorted.csv'
    data_set_file_name2 = 'Data/totest/febrl4_B_sorted.csv'

    exp = Experiment()
    exp.add_data_file(data_set_file_name1, 1)
    exp.add_data_file(data_set_file_name2, 2)

    attr_num_list = [1, 2, 3, 4, 5, 6, 7, 8]
    num_rec = [5000]
    sim_thres_list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    k = [10, 15, 20, 'opt']
    q = [2, 3]
    padded = [True, False]
    # (attr_num, q, padded, hash_method)

    hash_method = hashing.RandomHashingMethod()
    exp.set_hashing_method(hash_method)
    # exp.set_encoding_params(attr_num_list, 2, True, hash_method)

    #### Build pipelines ####
    c1 = encoding.CLKEncoding()
    b1 = blocking.HLSHBlocking(50)
    p1 = Pipeline(c1, b1)
    exp.add_pipeline(p1)

    c2 = encoding.RBFEncoding()
    b2 = blocking.MinHashBlocking(3,50)
    p2 = Pipeline(c2, b2)
    exp.add_pipeline(p2)

    exp.run(attr_num_list, sim_thres_list, num_rec, k, q, padded)


class Experiment:
    def __init__(self, with_q_gram=True):
        self.pipelines = []
        self.with_q_gram = with_q_gram
        self.set_reporting_options()
        self.report = Report()
        pass

    def set_hashing_method(self, h:HashingMethod):
        self.hash_method = h

    def set_report(self, r:Report):
        self.report = r

    def set_reporting_options(self, show_linkage_quality_report=True, show_blocking_report=True, show_plot=True, save_table=False):
        self.show_linkage_quality_report = show_linkage_quality_report
        self.show_blocking_report = show_blocking_report
        self.show_plot = show_plot
        self.save_table = save_table

    def add_pipeline(self, p:Pipeline):
        self.pipelines.append(p)

    def add_data_file(self, path, order):
        if order == 1:
            self.data_set_file_name1 = path
        else:
            self.data_set_file_name2 = path

    def load_data(self, attr_num_list,  num_rec, ent_id_col=0):
        # self.attr_num_list = attr_num_list
        max_attr_num = max(attr_num_list)
        rec_attr_val_dict1, header_list1 = load_data_set(self.data_set_file_name1, attr_num_list,
                                                         max_attr_num, ent_id_col, num_rec,
                                                         header_line_flag=True)

        rec_attr_val_dict2, header_list2 = load_data_set(self.data_set_file_name2, attr_num_list,
                                                         max_attr_num, ent_id_col, num_rec,
                                                         header_line_flag=True)
        return rec_attr_val_dict1, header_list1, rec_attr_val_dict2, header_list2

    def run(self, attr_num_list, sim_thres_list, num_rec_list, k_list=[10], q_list=[2], padd_list=[True]):
        self.report.init_blocking_report()

        for q in q_list:
            for padded in padd_list:
                for num_rec in num_rec_list:
                    for k in k_list:
                        self.run_config(attr_num_list, sim_thres_list, num_rec, k, q, padded)



    def run_config(self, attr_num_list, sim_thres_list, num_rec, k, q, padded, bf_len = 1024):
        true_matches_data_set = 'none'
        num_hash_funct = k
        non_matching=0
        conf_str='num_rec={}, non_matching_rows={}, q={}, padded={}, k={}, encode_methods={}, sim_thres={}'.format(num_rec, non_matching, q, padded, num_hash_funct, encode_type_list, sim_thres_list)
        print('>> '+conf_str)
        print('------------------------------------------------------------------------------------------------------------------------------\n')

        max_attr_num = max(attr_num_list)
        min_sim = min(sim_thres_list)
        rec_attr_val_dict1, header_list1, rec_attr_val_dict2, header_list2 = self.load_data(attr_num_list, num_rec)

        # Combine into one list for later use
        #
        all_rec_list = list(rec_attr_val_dict1.values()) + list(rec_attr_val_dict2.values())

        if true_matches_data_set != 'none':
            true_match_set = load_truth_data(true_matches_data_set)
        else:
            common_ent_id_set = set(rec_attr_val_dict1.keys()) & \
                                set(rec_attr_val_dict2.keys())
            true_match_set = set()
            for ent_id in common_ent_id_set:
                true_match_set.add((ent_id, ent_id))

        # Total number of record pair comparisons
        #
        all_comparisons = len(list(rec_attr_val_dict1.keys())) * \
                          len(list(rec_attr_val_dict2.keys()))

        if (num_hash_funct == 'opt'):
            num_hash_funct_str = 'opt'
            num_hash_funct = calc_opt(attr_num_list,all_rec_list,q,padded, bf_len)
        else:
            num_hash_funct_str = None

        self.hash_method.set_params(bf_len, num_hash_funct)

        # Initialize the encoding method
        #
        attr_encode_tuple_list = []
        for attr_num in attr_num_list:
            attr_tuple = (attr_num, q, padded, self.hash_method)
            attr_encode_tuple_list.append(attr_tuple)

        encode_type_list = []
        for pipeline in self.pipelines:
            pipeline.encode_method.set_params(attr_encode_tuple_list)
            encode_type_list.append(pipeline.encode_method.type)

        i=0

        # loop through submitted pipelines and run each of them.
        for pipeline in self.pipelines:
            # run the encoding step
            pipeline.run_encoding(rec_attr_val_dict1, header_list1, rec_attr_val_dict2, header_list2)
            # run the blocking step, record the time
            start_time = time.time()
            pipeline.run_blocking() # covers no-blocking setting
            blocking_time = time.time() - start_time

            # If this experiment test against clear-text q-gram, generate q-grams and conduct the linkage
            if self.with_q_gram and i==0:
                rec_q_gram_dict1 = gen_q_gram_dict(rec_attr_val_dict1, q, padded)
                rec_q_gram_dict2 = gen_q_gram_dict(rec_attr_val_dict2, q, padded)

                start_time = time.time()
                q_gram_rec_pair_dict = conduct_linkage(rec_q_gram_dict1, pipeline.block_dict1,
                                                       rec_q_gram_dict2, pipeline.block_dict2,
                                                       cal_q_gram_sim, min_sim)
                # record the time for q-gram linkage
                # time_dict['q-gram'] = time.time() - start_time
                self.report.add_time('q-gram', time.time() - start_time)
                q_gram_class_res_dict = calc_linkage_outcomes(q_gram_rec_pair_dict,
                                                              sim_thres_list, true_match_set,
                                                              all_comparisons)

            # conduct linkage for each pipeline encoded records, record the time
            start_time = time.time()
            bf_rec_pair_dict = pipeline.conduct_linkage(min_sim)
            bf_linkage_time = time.time() - start_time

            ####################
            # collecting results
            ####################
            blocking_statistics = pipeline.get_blocking_statistics(true_match_set)
            # meas = meas + [blocking_time, bf_linkage_time, blocking_statistics['num_matches'], blocking_statistics['cand_rec_num'], blocking_statistics['pc'], 1-(blocking_statistics['cand_rec_num']/(num_rec*num_rec))]
            mea = [blocking_time, bf_linkage_time, blocking_statistics['num_matches'], blocking_statistics['cand_rec_num'], blocking_statistics['pc'], 1-(blocking_statistics['cand_rec_num']/(num_rec*num_rec))]

            self.report.append_meas(mea)

            bf_class_res_dict = calc_linkage_outcomes(bf_rec_pair_dict,
                                                      sim_thres_list, true_match_set,
                                                      all_comparisons)
            print('Encoding method used:               %s' % (pipeline.encode_method.type))
            # time_dict[pipeline.encode_method.type] = bf_linkage_time

            self.report.add_time(pipeline.encode_method.type, bf_linkage_time)

            # Calulate precision and recall values for the different thresholds
            #
            if i == 0:
                self.report.init_report(sim_thres_list, q_gram_class_res_dict)

            self.report.update_report(sim_thres_list, bf_class_res_dict)

            i += 1

            ###############
            # Reporting
            ###############

            # Linkage time
            self.report.report_linkage_time()

            legend_str_list = ['Q-gram']
            for pipeline in self.pipelines:
                legend_str_list.append(pipeline.encode_method.type.upper())

            # display a table of linkage quality measures
            if self.show_linkage_quality_report:
                self.report.report_linkage_quality_meas(encode_type_list, sim_thres_list, conf_str)

            # plot figures of linkage quality measures
            if self.show_plot:
                self.report.plot_linkage_quality_fig(sim_thres_list,legend_str_list)

            # display a table of blocking measures
            if self.show_blocking_report:
                self.report.report_blocking_meas(num_rec)


if __name__ == "__main__":
    main()