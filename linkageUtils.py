# This module builds on the code provided here: 
# https://dmm.anu.edu.au/lsdbook2020/lsd_eval_programs-20201030.zip \   
        
# linkageUtils.py - Module that contains helper functions classes for the 
# evaluation framework
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

import csv
import gzip
import hashlib
import itertools
import math
import os
import random
import sys
import time
import binascii
import numpy
# For running on machines without display
#

from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison
import io
import pandas as pd

import matplotlib

# matplotlib.use('Agg')

PLOT_RATIO = 0.5

import matplotlib.pyplot as plt

# import encoding  # Bloom filter encoding module
# import hashing  # Bloom filter hashing module
# import hardening  # Bloom filter hardening module

DECIMAL_PLACES = 2  # Number of decimal places to use in rounding
PAD_CHAR = chr(1)  # Padding character to be added into record values

MAX_BLOCK_SIZE = 100  # Remove all blocks larger than this to limit memory use

ATTRIBUTES_PRINTED = False

def load_data_set(data_set_name, attr_num_list, max_attr_num, ent_id_col,
                  num_rec, col_sep_char=',', header_line_flag=False):
    """Load the data set to be used for linkage.

     It is assumed that the file to be loaded is a comma separated values file.

     Input arguments:
       - data_set_name      File name of the data set.
       - attr_num_list      The list of attributes to use (of the form [1,3]).
       - max_attr_num       The maximum attribute number to include.
       - ent_id_col         The column number of the entity identifiers.
       - num_rec            Number of records to be loaded from the file
                            If 'num_rec' is -1 all records will be read from
                            file, otherwise (assumed to be a positive integer
                            number) only the first 'num_rec' records will be
                            read.
       - col_sep_char       The column separate character.
       - header_line_flag   A flag, set this to True if the data set contains
                            a header line describing attributes, otherwise set
                            it to False. The default value False.

     Output:
       - rec_attr_val_dict  A dictionary with entity identifiers as keys and
                            where values are lists of attribute values.
  """

    rec_attr_val_dict = {}  # The dictionary of attribute value lists to be
    # loaded from file

    # Check if the file name is a gzip file or a csv file
    #
    if (data_set_name.endswith('gz')):
        in_f = gzip.open(data_set_name)
    else:
        in_f = open(data_set_name)

    # Initialise the csv reader
    #
    csv_reader = csv.reader(in_f, delimiter=col_sep_char)

    # Read the header line if available
    #
    header_list=''
    if (header_line_flag == True):
        header_list = next(csv_reader)

        # print('File header line:', header_list)
        global ATTRIBUTES_PRINTED
        if not ATTRIBUTES_PRINTED:
            print('  Attributes to be used:', end=' ')
            for attr_num in attr_num_list:
                print(header_list[attr_num], end=' ')
            print()
            ATTRIBUTES_PRINTED=True

    max_attr_num = max_attr_num + 1

    # Read each line in the file and store the required attribute values in a
    # list
    #
    rec_num = 0

    for rec_val_list in csv_reader:

        if (num_rec > 0) and (rec_num >= num_rec):
            break  # Read enough records

        rec_num += 1

        use_rec_val_list = []

        # Read the entity identifier
        # print(ent_id_col)
        # print(rec_val_list)

        ent_id = rec_val_list[ent_id_col].strip().lower()

        for attr_num in range(max_attr_num):

            if attr_num in attr_num_list:
                use_rec_val_list.append(rec_val_list[attr_num].lower().strip())
            else:
                use_rec_val_list.append('')

        rec_attr_val_dict[ent_id] = use_rec_val_list

        # if rec_num == 1 or rec_num == 2 or rec_num == 3: #>
        #     print('ent_id = ', ent_id, 'list = ', use_rec_val_list)

    in_f.close()

    # print('Loaded %d records from file' % (len(rec_attr_val_dict)))

    # print(list(rec_attr_val_dict.items())[:1])
    return rec_attr_val_dict, header_list


# -----------------------------------------------------------------------------
def load_data_set2(data_set_name, attr_num_list, max_attr_num, ent_id_col,
                   num_rec, file_order=1, size_of_nonmatch=1000, col_sep_char=',', header_line_flag=False):
    """Load the data set to be used for linkage.

     It is assumed that the file to be loaded is a comma separated values file.

     Input arguments:
       - data_set_name      File name of the data set.
       - attr_num_list      The list of attributes to use (of the form [1,3]).
       - max_attr_num       The maximum attribute number to include.
       - ent_id_col         The column number of the entity identifiers.
       - num_rec            Number of records to be loaded from the file
                            If 'num_rec' is -1 all records will be read from
                            file, otherwise (assumed to be a positive integer
                            number) only the first 'num_rec' records will be
                            read.
       - col_sep_char       The column separate character.
       - header_line_flag   A flag, set this to True if the data set contains
                            a header line describing attributes, otherwise set
                            it to False. The default value False.

     Output:
       - rec_attr_val_dict  A dictionary with entity identifiers as keys and
                            where values are lists of attribute values.
  """

    rec_attr_val_dict = {}  # The dictionary of attribute value lists to be
    # loaded from file

    # Check if the file name is a gzip file or a csv file
    #
    if (data_set_name.endswith('gz')):
        in_f = gzip.open(data_set_name)
    else:
        in_f = open(data_set_name)

    # Initialise the csv reader
    #
    csv_reader = csv.reader(in_f, delimiter=col_sep_char)

    # Read the header line if available
    #
    header_list = ''
    acc = []
    if (header_line_flag == True):
        header_list = next(csv_reader)
        acc.append(header_list[0])
        # print('File header line:', header_list)
        print('  Attributes to be used:', end=' ')

        for attr_num in attr_num_list:
            print(header_list[attr_num], end=' ')
            acc += [header_list[attr_num]]
        print()

    max_attr_num = max_attr_num + 1

    # Read each line in the file and store the required attribute values in a
    # list
    #
    rec_num = 0

    for rec_val_list in csv_reader:

        if (num_rec > 0) and (rec_num >= num_rec):
            break  # Read enough records

        rec_num += 1

        ####
        if rec_num >= num_rec - size_of_nonmatch:
            continue

        ####
        use_rec_val_list = []

        # Read the entity identifier
        # print(ent_id_col)
        # print(rec_val_list)

        ent_id = rec_val_list[ent_id_col].strip().lower()

        for attr_num in range(max_attr_num):

            if attr_num in attr_num_list:
                use_rec_val_list.append(rec_val_list[attr_num].lower().strip())
            else:
                use_rec_val_list.append('')

        rec_attr_val_dict[ent_id] = use_rec_val_list

        # if rec_num == 1 or rec_num == 2 or rec_num == 3: #>
        #     print('ent_id = ', ent_id, 'list = ', use_rec_val_list)

    if file_order == 1:
        for rec_val_list in csv_reader:

            if (num_rec > 0) and (rec_num >= num_rec + size_of_nonmatch):
                break  # Read enough records
            rec_num += 1

            #             print('file_order:', file_order, ', rec_num:', rec_num)

            use_rec_val_list = []
            ent_id = rec_val_list[ent_id_col].strip().lower()
            for attr_num in range(max_attr_num):
                if attr_num in attr_num_list:
                    use_rec_val_list.append(rec_val_list[attr_num].lower().strip())
                else:
                    use_rec_val_list.append('')
            rec_attr_val_dict[ent_id] = use_rec_val_list

    elif file_order == 2:
        for rec_val_list in csv_reader:

            if (num_rec > 0) and (rec_num >= num_rec + (2 * size_of_nonmatch)):
                break  # Read enough records

            rec_num += 1
            ####
            if rec_num <= num_rec + size_of_nonmatch:
                continue
                ####
            #             print('file_order:', file_order, ', rec_num:', rec_num)

            use_rec_val_list = []
            ent_id = rec_val_list[ent_id_col].strip().lower()
            for attr_num in range(max_attr_num):
                if attr_num in attr_num_list:
                    use_rec_val_list.append(rec_val_list[attr_num].lower().strip())
                else:
                    use_rec_val_list.append('')
            rec_attr_val_dict[ent_id] = use_rec_val_list

    in_f.close()

    # print('Loaded %d records from file' % (len(rec_attr_val_dict)))

    # print(list(rec_attr_val_dict.items())[:1])
    header_list = acc
    return rec_attr_val_dict, header_list




# -----------------------------------------------------------------------------
def load_truth_data(file_name):
    """Load a truth data file where each line contains two entity identifiers
     where the corresponding record pair is a true match.

     Input arguments:
       - file_name  The file name of the true matches. We assume the file
                    contains the entity identifier pairs of the true matches
                    of the two data sets that are being linked.

     Output:
       - truth_id_set  Returns a set where the elements are pairs (tuples)
                       of these record identifier pairs.
  """

    if (file_name.endswith('gz')):
        in_f = gzip.open(file_name, 'rt')
    else:
        in_f = open(file_name)

    csv_reader = csv.reader(in_f)

    print('Load truth data from file: ' + file_name)

    truth_id_set = set()

    for rec_list in csv_reader:
        assert len(rec_list) == 2, rec_list  # Make sure only two identifiers

        ent_id1 = rec_list[0].lower().strip()
        ent_id2 = rec_list[1].lower().strip()

        # Sort to make sure a record pair does not occur twice (i.e. not flipped)
        #
        true_match_pair_tuple = tuple(sorted([ent_id1, ent_id2]))
        truth_id_set.add(true_match_pair_tuple)

    in_f.close()

    print('  Loaded %d true matching record pairs' % (len(truth_id_set)))
    print('')

    return truth_id_set


# -----------------------------------------------------------------------------



def gen_bf_dict(rec_attr_val_dict, encoding_method, hardening_method=None,
                salt_attr_num=None):
    """Encode each of the given attribute value lists into one Bloom filter
     based on the given encoding method (and optionally harden them using the
     provided hardening method) and return a dictionary with Bloom filters
     as values.

     Input arguments:
       - rec_attr_val_dict  A list where each element is a list of attribute
                            values.
       - encoding_method    The method to be used to encode attribute values
                            into Bloom filters.
       - hardening_method   The (optional) method used to harden Bloom filters.
       - salt_attr_num      If the SALT hardening method is being used then
                            this parameter provides the attribute number to be
                            used for salting.

     Output:
       - rec_bf_dict  The dictionary of Bloom filters.
  """

    rec_bf_dict = {}  # The dictionary of Bloom filters.

    for (ent_id, use_rec_val_list) in rec_attr_val_dict.items():

        # If required apply salting or Markov chain hardening
        #
        if (hardening_method != None) and (hardening_method == 'SALT'):

            rec_salt_str = use_rec_val_list[salt_attr_num]

            # Check which encoding method is being used and generate a salt string
            # or salt string list accordingly
            #
            if (encoding_method.type == 'ABF'):
                rec_bf = encoding_method.encode(use_rec_val_list,
                                                salt_str=rec_salt_str)

            else:  # Need a list of salt strings
                rec_salt_str_list = []
                for i in range(len(use_rec_val_list)):
                    rec_salt_str_list.append(rec_salt_str)

                rec_bf = encoding_method.encode(use_rec_val_list,
                                                salt_str_list=rec_salt_str_list)

        elif (hardening_method != None) and (hardening_method.type == 'MC'):
            rec_bf = encoding_method.encode(use_rec_val_list,
                                            mc_harden_class=hardening_method)

        elif (hardening_method != None):
            rec_bf = encoding_method.encode(use_rec_val_list)
            rec_bf = hardening_method.harden_bf(rec_bf)

        else:  # No hardening
            rec_bf = encoding_method.encode(use_rec_val_list)

        # Add the generated Bloom filters into the dictionary
        #
        rec_bf_dict[ent_id] = rec_bf

    # print('Generated %d Bloom filters' % (len(rec_bf_dict)))

    return rec_bf_dict


# -----------------------------------------------------------------------------

def gen_bf_dict_external(rec_attr_val_dict, header_list, attr_num_list, bf_length, n_gram, num_of_hash_funcs, bits_per_feature=False):
    print('rec_attr_val_dict:')

    header_list=['empty']+header_list[1:]

    # for k,v in rec_attr_val_dict.items():
    #     print(rec_attr_val_dict[k])
    #     break
    #
    # print('passed header:')
    # print(header_list)

    copy_h = header_list.copy()
    copy_h_index = 1
    header_list = ['empty']
    prev_num = 0
    del_acc=[]
    for num in attr_num_list:

        if (num - prev_num) != 1:

            for i in range(prev_num+1, num):

                header_list+=['del'+str(i)]
                del_acc+=['del'+str(i)]

        header_list+=[copy_h[copy_h_index]]
        copy_h_index+=1
        prev_num = num

    df = pd.DataFrame.from_dict(rec_attr_val_dict, orient='index', columns=header_list)
    del df['empty']
    for val in del_acc:
        del df[val]
    header_list = copy_h
    df.index.name='ncid'
    csv_file = io.StringIO()
    df.to_csv(csv_file)
    bits_per_att = int(bf_length / (len(header_list)-1))
    fields = [
        Ignore('ncid')
    ]
    for att in header_list[1:]:
        if bits_per_feature:
            fields.append(StringSpec(att, FieldHashingProperties(comparator=NgramComparison(n_gram), strategy=BitsPerFeatureStrategy(bits_per_att))))
        else:
            fields.append(StringSpec(att, FieldHashingProperties(comparator=NgramComparison(n_gram), strategy=BitsPerTokenStrategy(num_of_hash_funcs))))
    print('bf_length =', bf_length)
    schema = Schema(fields, bf_length)
    secret='secret'
    csv_file.seek(0)
    clk_rbf = clk.generate_clk_from_csv(csv_file, secret, schema, progress_bar=False)
    res={}
    i=0
    for k in rec_attr_val_dict:
        res[k]=clk_rbf[i]
        i+=1
    return res

# -----------------------------------------------------------------------------

def gen_q_gram_dict(rec_attr_val_dict, q, padded=False):
    """Generate a set of q-grams for each of the given attribute value lists
     and return a dictionary with q-gram sets as values.

     Input arguments:
       - rec_attr_val_dict  A list where each element is a list of attribute
                            values.
       - q                  The length of a q-gram.
       - padded             A flag to set if an extra character needs to be
                            added into the front and back of each attribute
                            value.
     Output:
       - rec_q_gram_dict  The dictionary of q-gram sets.
  """

    rec_q_gram_dict = {}  # The dictionary of q-gram sets.

    qm1 = q - 1
    toggle = True
    for (ent_id, use_rec_val_list) in rec_attr_val_dict.items():

        q_gram_set = set()

        for attr_val in use_rec_val_list:

            if (padded == True):
                attr_val = PAD_CHAR * qm1 + attr_val + PAD_CHAR * qm1

            attr_val_len = len(attr_val)

            # Generate set of q-grams for the attribute value
            #
            attr_q_gram_set = set([attr_val[i:i + q]
                                   for i in range(attr_val_len - qm1)])

            # Add the attribute q-gram set into the record q-gram set
            #
            q_gram_set = q_gram_set.union(attr_q_gram_set)

        # Add the generated q-gram set into the dictionary
        #
        rec_q_gram_dict[ent_id] = q_gram_set
        if toggle:
            # print('q grams ============= > > > > >')
            # print(q_gram_set)
            toggle = False
    # print('Generated %d q-gram sets' % (len(rec_q_gram_dict)))

    return rec_q_gram_dict


# -----------------------------------------------------------------------------

def gen_q_gram_blk_dict(rec_attr_val_dict, q, attr_list, padded=False):
    """Generate a set of q-grams for each of the given attribute value lists
     and return a dictionary with q-gram sets as values.

     Input arguments:
       - rec_attr_val_dict  A list where each element is a list of attribute
                            values.
       - q                  The length of a q-gram.
       - padded             A flag to set if an extra character needs to be
                            added into the front and back of each attribute
                            value.
     Output:
       - rec_q_gram_dict  The dictionary of q-gram sets.
  """

    rec_q_gram_dict = {}  # The dictionary of q-gram sets.

    qm1 = q - 1

    for (ent_id, use_rec_val_list) in rec_attr_val_dict.items():

        q_gram_set = set()
        i = 0
        for attr_val in use_rec_val_list:
            if i in attr_list:
                if (padded == True):
                    attr_val = PAD_CHAR * qm1 + attr_val + PAD_CHAR * qm1

                attr_val_len = len(attr_val)

                # Generate set of q-grams for the attribute value
                #
                attr_q_gram_set = set([attr_val[i:i + q]
                                       for i in range(attr_val_len - qm1)])

                # Add the attribute q-gram set into the record q-gram set
                #
                q_gram_set = q_gram_set.union(attr_q_gram_set)
            i += 1
        # Add the generated q-gram set into the dictionary
        #
        rec_q_gram_dict[ent_id] = q_gram_set

    # print('Generated %d q-gram sets' % (len(rec_q_gram_dict)))

    return rec_q_gram_dict


# -----------------------------------------------------------------------------

def cal_q_gram_sim(q_gram_set1, q_gram_set2):
    """Calculate the Dice Similarity between the two given q-gram sets.

     Dice similarity is calculated between two sets A and B as

        Dice similarity (A,B) = 2 x number of common elements of A and B
                                -------------------------------------------
                                number of elems in A + number of elems in B

     Input arguments:
       - q_gram_set1  The first q-gram set.
       - q_gram_set2  The second q-gram set.

     Output:
       - q_gram_sim  The dice similarity between the two q-gram sets.
  """

    num_q_gram_common = len(q_gram_set1.intersection(q_gram_set2))

    q_gram_sim = (2.0 * num_q_gram_common) / (float(len(q_gram_set1)) + \
                                              float(len(q_gram_set2)))

    return q_gram_sim


# -----------------------------------------------------------------------------

def cal_bf_sim(bf1, bf2):
    """Calculate the Dice Similarity between the two given Bloom filters.

     Dice similarity is calculated between two sets A and B as

        Dice similarity (A,B) = 2 x number of common elements of A and B
                                -------------------------------------------
                                number of elems in A + number of elems in B

     Input arguments:
       - bf1  The first Bloom filter.
       - bf2  The second Bloom filter.

     Output:
       - bf_sim  The dice similarity between the Bloom filter pair.
  """

    assert len(bf1) == len(bf2)

    num_ones_bf1 = bf1.count(1)
    num_ones_bf2 = bf2.count(1)

    bf_common = bf1 & bf2
    num_common_ones = bf_common.count(1)
    bf_sim = (2 * num_common_ones) / (float(num_ones_bf1) + \
                                      float(num_ones_bf2))

    return bf_sim


# -----------------------------------------------------------------------------
# --------------------------------- BLOCKING ----------------------------------
# -----------------------------------------------------------------------------
def init_minhash(lsh_band_size, lsh_num_band):
    """Initialise the parameters for Min-Hash Locality Sensitive Hashing (LSH)
     including generating random values for hash functions.

     LSH min-hashing follows the code provided here:
     https://github.com/chrisjmccormick/MinHash/blob/master/ \
               runMinHashExample.py

     The probability for a pair of sets with Jaccard sim 0 < s <= 1 to be
     included as a candidate pair is (with b = lsh_num_band and
     r = lsh_band_size, i.e. the number of rows/hash functions per band) is
     (Leskovek et al., 2014):

     p_cand = 1- (1 - s^r)^b

     Approximation of the 'threshold' of the S-curve (Leskovek et al., 2014)
     is: t = (1/k)^(1/r)

     If a string is given as plot_file_name then a graph of the probabilities
     will be generated and saved into this file.
  """

    # Calculate error probabilities for given parameter values
    #
    assert lsh_num_band > 1, lsh_num_band
    assert lsh_band_size > 1, lsh_band_size

    num_hash_funct = lsh_band_size * lsh_num_band  # Total number needed

    b = float(lsh_num_band)
    r = float(lsh_band_size)
    t = (1.0 / b) ** (1.0 / r)

    s_p_cand_list = []
    for i in range(1, 10):
        s = 0.1 * i
        p_cand = 1.0 - (1.0 - s ** r) ** b
        assert 0.0 <= p_cand <= 1.0
        s_p_cand_list.append((s, p_cand))

    print()
    print('Initialise LSH blocking using Min-Hash')
    print('  Number of hash functions: %d' % (num_hash_funct))
    print('  Number of bands:          %d' % (lsh_num_band))
    print('  Size of bands:            %d' % (lsh_band_size))
    print('  Threshold of s-curve:     %.3f' % (t))
    print('  Probabilities for candidate pairs:')
    print('   Jacc_sim | prob(cand)')
    for (s, p_cand) in s_p_cand_list:
        print('     %.2f   |   %.5f' % (s, p_cand))
    print()

    max_hash_val = 2 ** 31 - 1  # Maximum possible value a CRC hash could have

    # Random hash function will take the form of: h(x) = (a*x + b) % c
    # where 'x' is the input value, 'a' and 'b' are random
    # coefficients, and
    # 'c' is a prime number just greater than max_hash_val
    #
    # Generate 'num_hash_funct' coefficients
    #
    coeff_a_set = set()
    coeff_b_set = set()

    while (len(coeff_a_set) < num_hash_funct):
        coeff_a_set.add(random.randint(0, max_hash_val))
    while (len(coeff_b_set) < num_hash_funct):
        coeff_b_set.add(random.randint(0, max_hash_val))
    coeff_a_list = sorted(coeff_a_set)
    coeff_b_list = sorted(coeff_b_set)
    assert coeff_a_list != coeff_b_list

    return coeff_a_list, coeff_b_list


# ---------------------------------------------------------------------------

def hash_q_gram_set(q_gram_set, coeff_a_list, coeff_b_list,
                    lsh_band_size, lsh_num_band):
    """Min-hash the given set of q-grams and return a list of hash signatures
     depending upon the Min-hash parameters set with the 'init_minhash'
     method.
  """

    # We need the next largest prime number above 'maxShingleID'.
    # From here:
    # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    #
    next_prime = 4294967311

    crc_hash_set = set()

    for q_gram in q_gram_set:  # Hash the q-grams into 32-bit integers
        crc_hash_set.add(binascii.crc32(bytes(q_gram, encoding="ascii")) & 0xffffffff)

    assert len(q_gram_set) == len(crc_hash_set)  # Check no collision

    # Now generate all the min-hash values for this q-gram set
    #
    min_hash_sig_list = []
    num_hash_funct = lsh_band_size * lsh_num_band
    for h in range(num_hash_funct):

        # For each CRC hash value (q-gram) in the q-gram set calculate
        # its Min-hash value for all 'num_hash_funct' functions
        #
        min_hash_val = next_prime + 1  # Initialise to value outside range

        for crc_hash_val in crc_hash_set:
            hash_val = (coeff_a_list[h] * crc_hash_val + coeff_b_list[h]) % \
                       next_prime
            min_hash_val = min(min_hash_val, hash_val)

        min_hash_sig_list.append(min_hash_val)

    # Now split hash values into bands and generate the list of
    # 'lsh_num_band' hash values used for blocking
    #
    band_hash_sig_list = []

    start_ind = 0
    end_ind = lsh_band_size

    for band_num in range(lsh_num_band):
        band_hash_sig = min_hash_sig_list[start_ind:end_ind]
        assert len(band_hash_sig) == lsh_band_size, len(band_hash_sig)
        start_ind = end_ind
        end_ind += lsh_band_size
        band_hash_sig_list.append(band_hash_sig)

    return band_hash_sig_list


# ---------------------------------------------------------------------------
def minhash_blocking(q_gram_set_dict, coeff_a_list, coeff_b_list,
                     lsh_band_size, lsh_num_band, max_block_size):
    """Block the individual records using the min-hash LSH algorithm, which
     extracts q-grams from values in the attribute lists given in the
     'lsh_block_def_dict', and represents each individual by a
     signature of hashed q-grams.

     The 'lsh_block_def_dict' must contain keys made of record id
     and corresponding values of attributes.

     If a key is None then all individual records will be hashed.

     The method returns a dictionary with min-hash buckets (as
     blocking key values) as keys and sets of identifiers as values.
  """

    print('Conduct Min-hash LSH blocking')
    print()

    start_time = time.time()

    minhash_dict = {}  # Keys will be min-hash signatures, values set of
    # identifiers

    num_empty_q_gram_set = 0
    num_rec_hashed = 0

    # Keep a set of all individuals to be hashed so we can check if all are
    # included in a block
    #
    all_hashed_ind_id_set = set()

    # Loop over all individuals, and if their certificate and role types
    # fulfill one of the given ones hash the individual's record
    #
    for (rec_id, q_gram_set) in q_gram_set_dict.items():

        num_rec_hashed += 1

        if (len(q_gram_set) == 0):
            num_empty_q_gram_set += 1
            continue  # Do not index empty q-gram set

        # Get the min-hash signatures of this q-gram set
        #

        band_hash_sig_list = hash_q_gram_set(q_gram_set, coeff_a_list,
                                             coeff_b_list, lsh_band_size, lsh_num_band)


        assert len(band_hash_sig_list) == lsh_num_band

        # Insert each individual into blocks according to its min-hash values
        #

        for band_hash_sig in band_hash_sig_list:
            # To save memory convert into a MD5 hashes
            #
            block_key_val = hashlib.md5(str(band_hash_sig).encode('ascii')).digest()
            # block_key_val = str(band_hash_sig)

            block_rec_id_set = minhash_dict.get(block_key_val, set())
            block_rec_id_set.add(rec_id)
            minhash_dict[block_key_val] = block_rec_id_set

    assert len(q_gram_set_dict) == num_rec_hashed

    print('  Hashed %d records in %.2f sec' % (num_rec_hashed,
                                               time.time() - start_time))
    print('    Inserted each record into %d blocks' % (lsh_num_band))
    print('      Number of empty attribute value q-gram sets: %d' % \
          (num_empty_q_gram_set))

    # Calculate statistics of generated blocks
    #
    block_size_list = []
    num_pair_comp = 0  # How many comparisons to be done with this blocking
    num_block_size1 = 0  # Number of blocks with only one individual
    all_block_size_list = []
    num_large_block_del = 0

    for block_key in list(minhash_dict.keys()):
        block_ent_id_set = minhash_dict[block_key]
        num_rec_in_block = len(block_ent_id_set)
        all_block_size_list.append(num_rec_in_block)
        if (num_rec_in_block > max_block_size):
            del minhash_dict[block_key]
            num_large_block_del += 1
        else:
            block_size_list.append(num_rec_in_block)

    print('  Minimum, average and maximum block sizes (all blocks): ' + \
          '%d / %.2f / %d' % (min(all_block_size_list),
                              float(sum(all_block_size_list)) / len(all_block_size_list),
                              max(all_block_size_list)))
    print('    Removed %d blocks larger than %d records, %d blocks left' % \
          (num_large_block_del, max_block_size, len(block_size_list)))
    if (len(block_size_list) > 0):
        print('    Minimum, average and maximum block sizes: %d / %.2f / %d' % \
              (min(block_size_list), float(sum(block_size_list)) / \
               len(block_size_list), max(block_size_list)))
    else:
        print('    Warning: No blocks left')
    print()

    del block_size_list
    return minhash_dict


# -----------------------------------------------------------------------------

def hlsh_blocking(rec_bf_dict, num_seg, max_block_size):
    """Split the given dictionary of records into smaller blocks based on Bloom
     filter sets of bit position of length 'num_bit_pos_key' they have in
     common and return a blocking dictionary.

     Input arguments:
       - rec_bf_dict     A dictionary of records with their entity identifiers
                         as keys and corresponding Bloom filters as values.
       - num_seg         The number of segments of Bloom filter bit positions
                         to be used to generate blocking keys.
       - max_block_size  The maximum number of records allowed in a block, all
                         larger blocks will be removed (to limit memory use
                         and run times).

     Output:
       - block_dict  A dictionary with blocking keys as keys and sets of entity
                     identifiers as values.
  """

    assert num_seg >= 1, num_seg
    assert max_block_size > 1, max_block_size

    block_dict = {}

    bf_len = None

    # Loop over all records and extract all Bloom filter bit position sub arrays
    # of length 'num_bit_pos_key' and insert the record into these corresponding
    # blocks
    #
    for (ent_id, rec_bf) in rec_bf_dict.items():

        # First time calculate the indices to use for splitting a Bloom filter
        #
        if (bf_len == None):
            bf_len = len(rec_bf)

            seg_len = int(float(bf_len) / num_seg)

            bf_split_index_list = []
            start_pos = 0
            end_pos = seg_len
            while (end_pos <= bf_len):
                bf_split_index_list.append((start_pos, end_pos))
                start_pos = end_pos
                end_pos += seg_len

            # Depending upon the Bloom filter length and 'num_bit_pos_key' to use
            # the last segement might contain less than 'num_bit_pos_key' positions.

        # Extract the bit position arrays for these segments
        #
        for (start_pos, end_pos) in bf_split_index_list:
            bf_seg = rec_bf[start_pos:end_pos]

            block_key = bf_seg.to01()  # Make it a string
            block_ent_id_set = block_dict.get(block_key, set())
            block_ent_id_set.add(ent_id)
            block_dict[block_key] = block_ent_id_set


    # print('Bloom filter HLSH blocking dictionary contains %d blocks ' % \
    #       (len(block_dict)) + ' (with %d segments)' % (num_seg))


    # print('  Segments:', bf_split_index_list)
    block_size_list = []
    all_block_size_list = []
    num_large_block_del = 0

    for block_key in list(block_dict.keys()):
        block_ent_id_set = block_dict[block_key]
        num_rec_in_block = len(block_ent_id_set)
        all_block_size_list.append(num_rec_in_block)
        if (num_rec_in_block > max_block_size):
            del block_dict[block_key]
            num_large_block_del += 1
        else:
            block_size_list.append(num_rec_in_block)
    # print('  Minimum, average and maximum block sizes (all blocks): ' + \
    #       '%d / %.2f / %d' % (min(all_block_size_list),
    #                           float(sum(all_block_size_list)) / len(all_block_size_list),
    #                           max(all_block_size_list)))
    # print('    Removed %d blocks larger than %d records, %d blocks left' % \
    #       (num_large_block_del, max_block_size, len(block_size_list)))
    if (len(block_size_list) > 0):
        pass
        # print('    Minimum, average and maximum block sizes: %d / %.2f / %d' % \
        #       (min(block_size_list), float(sum(block_size_list)) / \
        #        len(block_size_list), max(block_size_list)))
    else:
        print('    Warning: No blocks left')
    print()

    return block_dict

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def conduct_linkage(val_dict1, block_dict1, val_dict2, block_dict2, sim_funct,
                    min_sim):
    """Compare records from the given two dictionaries with blocks of records and
     store those pairs with a similarity of at least 'min_sim'.

     Input arguments:
       - val_dict1    First dictionary of records with their entity identifiers
                      as keys and corresponding values to be compared.
       - block_dict1  First dictionary of blocks where keys are some form
                      block representation while values are sets of entity
                      identifiers in this block.
       - val_dict2    Second dictionary of records with their entity
                      identifiers as keys and corresponding values to be
                      compared.
       - block_dict1  Second dictionary of blocks where keys are some form
                      block representation while values are sets of entity
                      identifiers in this block.
       - sim_funct    The function to be used for similarity calculation.
       - min_sim      The minimum similarity of records pairs so they are
                      included in the results dictionary to be returned.

     Output:
       - rec_pair_dict  A dictionary with the compared record pairs (their
                        entity identifiers as keys) that have a similarity of
                        at least 'min_sim' (as values).
  """

    assert min_sim >= 0.0 and min_sim <= 1.0, min_sim

    rec_pair_dict = {}

    # Keep track of pairs compared so each pair is only compared once
    #
    pairs_compared_set = set()

#     print('Similarity threshold based classification')
#     print('  Minimum similarity of record pairs to be stored: %.2f' % (min_sim))

    # Iterate over all block values that occur in both data sets
    #
    for block_val1 in block_dict1.keys():

        if (block_val1 not in block_dict2):
            continue  # Block value not in common, go to next one

        ent_id_set1 = block_dict1[block_val1]
        ent_id_set2 = block_dict2[block_val1]

        # Iterate over all value pairs
        #
        for ent_id1 in ent_id_set1:
            val1 = val_dict1[ent_id1]

            for ent_id2 in ent_id_set2:

                ent_id_pair = (ent_id1, ent_id2)

                if (ent_id_pair not in pairs_compared_set):

                    val2 = val_dict2[ent_id2]

                    pairs_compared_set.add(ent_id_pair)

                    sim = sim_funct(val1, val2)  # Calculate the similarity

                    if (sim >= min_sim):
                        rec_pair_dict[ent_id_pair] = sim

    num_all_comparisons = len(val_dict1) * len(val_dict2)
    num_pairs_compared = len(pairs_compared_set)

#     print('Compared %d record pairs (full comparison is %d record pairs)' % \
#           (num_pairs_compared, num_all_comparisons))
#     print('  Reduction ratio: %2f' % (1.0 - float(num_pairs_compared) / \
#                                       num_all_comparisons))
#     print('  Stored %d record pairs with a similarity of at least %.2f' % \
#           (len(rec_pair_dict), min_sim))
#     print()

    pairs_compared_set = set()

    return rec_pair_dict



# -----------------------------------------------------------------------------

def calc_linkage_outcomes(rec_pair_dict, sim_thres_list, true_match_set,
                          num_all_comp):
    """Calculate the number of true positives, false positives, true negatives,
     and false negatives for the given record pair dictionary and list of
     similarity thresholds.

     Input arguments:
       - rec_pair_dict   A dictionary with the compared record pairs (their
                         entity identifiers as keys) and their calculated
                         similarities.
       - sim_thres_list  A list of classification similarity thresholds.
       - true_match_set  Set of true matches (entity identifier pairs).
       - num_all_comp    The total number of comparisons between all record
                         pairs.

     Output:
       - class_res_dict  A dictionary with similarity thresholds as keys and
                         for each threshold a tuple with the numbers of TP, FP,
                         TN and FN at that similarity threshold.
  """

    class_res_dict = {}

    for sim_thres in sim_thres_list:
        assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

        class_res_dict[sim_thres] = [0, 0, 0, 0]  # Initalise the results counters

    for (ent_id_pair, sim) in rec_pair_dict.items():

        if (ent_id_pair in true_match_set):
            is_true_match = True
        else:
            is_true_match = False

        #  Calculate linkage results for all given similarities
        #
        for sim_thres in sim_thres_list:
            sim_res_list = class_res_dict[sim_thres]

            if (sim >= sim_thres) and (is_true_match == True):  # TP
                sim_res_list[0] += 1
            elif (sim >= sim_thres) and (is_true_match == False):  # FP
                sim_res_list[1] += 1
            elif (sim < sim_thres) and (is_true_match == True):  # FN
                sim_res_list[3] += 1
            # TN are calculated at the end

    for sim_thres in sim_thres_list:
        sim_res_list = class_res_dict[sim_thres]

        # Calculate the number of TN as all comparisons - (TP + FP + FN)
        #
        sim_res_list[2] = num_all_comp - sum(sim_res_list)

        assert sum(sim_res_list) == num_all_comp

    return class_res_dict


# -----------------------------------------------------------------------------

def calc_precision(num_tp, num_fp, num_fn, num_tn):
    """Compute precision using the given confusion matrix.

     Precision is calculated as TP / (TP + FP).

     Input arguments:
       - num_tp  Number of true positives.
       - num_fp  Number of false positives.
       - num_fn  Number of false negatives.
       - num_tn  Number of true negatives.

     Output:
       - precision  A float value between 0 and 1.
  """

    if ((num_tp + num_fp) > 0):
        precision = float(num_tp) / (num_tp + num_fp)
    else:
        precision = 0.0

    return precision


# -----------------------------------------------------------------------------

def calc_recall(num_tp, num_fp, num_fn, num_tn):
    """Compute recall using the given confusion matrix.

     Recall is calculated as TP / (TP + FN).

     Input arguments:
       - num_tp  Number of true positives.
       - num_fp  Number of false positives.
       - num_fn  Number of false negatives.
       - num_tn  Number of true negatives.

     Output:
       - recall  A float value between 0 and 1.
  """

    if ((num_tp + num_fn) > 0):
        recall = float(num_tp) / (num_tp + num_fn)
    else:
        recall = 0.0

    return recall


# -----------------------------------------------------------------------------

def calc_fmeasure(num_tp, num_fp, num_fn, num_tn):
    if num_tp == 0:
        return 0
    return num_tp/(num_tp+(num_fp+num_fn)/2)

# -----------------------------------------------------------------------------

# We do not provide a function to calculate the F-measure given recent research
# has shown problematic issues with the F-measure when used to compare
# classifers for record linkage, see:
#
# A note on using the F-measure for evaluating record linkage algorithms
# David Hand and Peter Christen
# Statistics and Computing, Volume 28, Issue 3, pp 539-547, 2018.

# -----------------------------------------------------------------------------

def draw_plot(qg_prec_list, qg_reca_list, enc_prec_list, enc_reca_list,
              x_label, y_label, legend_str_list, title1, title2=None,
              aspect_ratio=0.35, x_min=0, x_max=1, y_min=0, y_max=1,
              title_font_size=30, label_font_size=28, axis_font_size=26,
              margin=0.05, plot_type='eps', save_fig_name=None):
    """Plot the two given lists of precision and recall values as a
     precision-recall graph (precision on the vertical and recall on the
     horizontal axis), and either saving the plot into a file or showing the
     plot and wait for enter to be pressed.

     Input arguments:
       -qg_prec_list      A list of precision values obtained when linking
                          q-gram representations of records.
       -qg_reca_list      A list of recall values obtained when linking q-gram
                          representations of records.
       -enc_prec_list     A list of precision values obtained when linking
                          Bloom filter representations of records.
       -enc_reca_list     A list of recall values obtained when linking Bloom
                          filter representations of records.
       - x_label          x axis label.
       - y_label          y axis label.
       - legend_str_list  List of string to be used for the legend.
       - title1           Main title of the plot.
       - title2           Secondary title of the plot.
       - aspect_ratio     The aspect ratio of the plot.
       - x_min            Minimum x value.
       - x_max            Maximum x value.
       - y_min            Minimum y value.
       - y_max            Maximum y value.
       - title_font_size  Font size of the titles of the plot.
       - label_font_size  Font size of the labels of the plot.
       - axis_font_size   Font size of the axis labels of the plot.
       - margin           The size of the margin around the plot.
       - plot_type        The file format of the plot.
       - save_fig_name    The file name for saving the plot, or None if the
                          plot is to be shown.

     Output:
       - This method does not return anything.
  """

    assert len(qg_prec_list) == len(qg_reca_list)
    for i in range(len(enc_prec_list)):
        assert len(enc_prec_list[i]) == len(enc_reca_list[i])

    # Change the aspect ratio of the plot
    #
    w, h = plt.figaspect(aspect_ratio)
    plt.figure(figsize=(w, h))
    matplotlib.rcParams.update({'font.size': axis_font_size})
    # Add the label for the x-axis and y-axis
    #
    if (x_label != None):
        plt.xlabel(x_label) #, fontsize=label_font_size)
    if (y_label != None):
        plt.ylabel(y_label) #, fontsize=label_font_size)

    # Plot the two precision-recall lines
    #
    line_style_list = ['-', '--', '-.', ':']
    color_list = ['#ff0000', '#0000ff', '#000000', '#FFFACD']

    for i in range(len(enc_prec_list) + 1):

        line_style_index = i % len(line_style_list)
        color_index = i % len(color_list)
        if i == 0:
            if (len(legend_str_list) > 0):
                plt.plot(qg_reca_list, qg_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index],
                         label=legend_str_list[0])
            else:
                plt.plot(qg_reca_list, qg_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index])
        else:
            bf_reca_list = enc_reca_list[i - 1]
            bf_prec_list = enc_prec_list[i - 1]
            if (len(legend_str_list) > 0):
                plt.plot(bf_reca_list, bf_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index],
                         label=legend_str_list[i])
            else:
                plt.plot(bf_reca_list, bf_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index])

    # Set the axis minimum and maximum values
    #
    plt.xlim(xmin=x_min - margin, xmax=x_max + margin)
    plt.ylim(ymin=y_min - margin, ymax=y_max + margin)

    # Add the title(s) into the plot
    #
    if (title2 != None or title2 != ''):
        # plt.suptitle(title2, y=1.0, fontsize=title_font_size)
        plt.title(title1)#, fontsize=title_font_size)
    else:
        plt.title(title1)#, fontsize=title_font_size)

    plt.legend(loc='best')#, fontsize=axis_font_size)

    if (save_fig_name != None):
        plt.savefig(save_fig_name, bbox_inches='tight')
        plt.clf()

    else:
        input('Press return')
        plt.clf()

#
#
#
#
#
#
def calc_linkage_outcomes(rec_pair_dict, sim_thres_list, true_match_set,
                          num_all_comp):
    """Calculate the number of true positives, false positives, true negatives,
     and false negatives for the given record pair dictionary and list of
     similarity thresholds.

     Input arguments:
       - rec_pair_dict   A dictionary with the compared record pairs (their
                         entity identifiers as keys) and their calculated
                         similarities.
       - sim_thres_list  A list of classification similarity thresholds.
       - true_match_set  Set of true matches (entity identifier pairs).
       - num_all_comp    The total number of comparisons between all record
                         pairs.

     Output:
       - class_res_dict  A dictionary with similarity thresholds as keys and
                         for each threshold a tuple with the numbers of TP, FP,
                         TN and FN at that similarity threshold.
  """

    class_res_dict = {}

    for sim_thres in sim_thres_list:
        assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

        class_res_dict[sim_thres] = [0, 0, 0, 0]  # Initalise the results counters

    for (ent_id_pair, sim) in rec_pair_dict.items():

        if (ent_id_pair in true_match_set):
            is_true_match = True
        else:
            is_true_match = False

        #  Calculate linkage results for all given similarities
        #
        for sim_thres in sim_thres_list:
            sim_res_list = class_res_dict[sim_thres]

            if (sim >= sim_thres) and (is_true_match == True):  # TP
                sim_res_list[0] += 1
            elif (sim >= sim_thres) and (is_true_match == False):  # FP
                sim_res_list[1] += 1
            elif (sim < sim_thres) and (is_true_match == True):  # FN
                sim_res_list[3] += 1
            # TN are calculated at the end

    for sim_thres in sim_thres_list:
        sim_res_list = class_res_dict[sim_thres]

        # Calculate the number of TN as all comparisons - (TP + FP + FN)
        #
        sim_res_list[2] = num_all_comp - sum(sim_res_list)

        assert sum(sim_res_list) == num_all_comp

    return class_res_dict


# -----------------------------------------------------------------------------

def calc_precision(num_tp, num_fp, num_fn, num_tn):
    """Compute precision using the given confusion matrix.

     Precision is calculated as TP / (TP + FP).

     Input arguments:
       - num_tp  Number of true positives.
       - num_fp  Number of false positives.
       - num_fn  Number of false negatives.
       - num_tn  Number of true negatives.

     Output:
       - precision  A float value between 0 and 1.
  """

    if ((num_tp + num_fp) > 0):
        precision = float(num_tp) / (num_tp + num_fp)
    else:
        precision = 0.0

    return precision


# -----------------------------------------------------------------------------

def calc_recall(num_tp, num_fp, num_fn, num_tn):
    """Compute recall using the given confusion matrix.

     Recall is calculated as TP / (TP + FN).

     Input arguments:
       - num_tp  Number of true positives.
       - num_fp  Number of false positives.
       - num_fn  Number of false negatives.
       - num_tn  Number of true negatives.

     Output:
       - recall  A float value between 0 and 1.
  """

    if ((num_tp + num_fn) > 0):
        recall = float(num_tp) / (num_tp + num_fn)
    else:
        recall = 0.0

    return recall

# -----------------------------------------------------------------------------

def draw_plot(qg_prec_list, qg_reca_list, enc_prec_list, enc_reca_list,
              x_label, y_label, legend_str_list, title1, title2=None,
              aspect_ratio=0.35, x_min=0, x_max=1, y_min=0, y_max=1,
              title_font_size=30, label_font_size=28, axis_font_size=26,
              margin=0.05, plot_type='eps', save_fig_name=None):
    """Plot the two given lists of precision and recall values as a
     precision-recall graph (precision on the vertical and recall on the
     horizontal axis), and either saving the plot into a file or showing the
     plot and wait for enter to be pressed.

     Input arguments:
       -qg_prec_list      A list of precision values obtained when linking
                          q-gram representations of records.
       -qg_reca_list      A list of recall values obtained when linking q-gram
                          representations of records.
       -enc_prec_list     A list of precision values obtained when linking
                          Bloom filter representations of records.
       -enc_reca_list     A list of recall values obtained when linking Bloom
                          filter representations of records.
       - x_label          x axis label.
       - y_label          y axis label.
       - legend_str_list  List of string to be used for the legend.
       - title1           Main title of the plot.
       - title2           Secondary title of the plot.
       - aspect_ratio     The aspect ratio of the plot.
       - x_min            Minimum x value.
       - x_max            Maximum x value.
       - y_min            Minimum y value.
       - y_max            Maximum y value.
       - title_font_size  Font size of the titles of the plot.
       - label_font_size  Font size of the labels of the plot.
       - axis_font_size   Font size of the axis labels of the plot.
       - margin           The size of the margin around the plot.
       - plot_type        The file format of the plot.
       - save_fig_name    The file name for saving the plot, or None if the
                          plot is to be shown.

     Output:
       - This method does not return anything.
  """

    assert len(qg_prec_list) == len(qg_reca_list)
    for i in range(len(enc_prec_list)):
        assert len(enc_prec_list[i]) == len(enc_reca_list[i])

    # Change the aspect ratio of the plot
    #
    w, h = plt.figaspect(aspect_ratio)
    plt.figure(figsize=(w, h))
    # matplotlib.rcParams.update({'font.size': axis_font_size})
    # Add the label for the x-axis and y-axis
    #
    if (x_label != None):
        plt.xlabel(x_label) #, fontsize=label_font_size)
    if (y_label != None):
        plt.ylabel(y_label) #, fontsize=label_font_size)

    # Plot the two precision-recall lines
    #
    line_style_list = ['-', '--', '-.', ':']
    color_list = ['#ff0000', '#0000ff', '#000000', '#FFFACD']

    for i in range(len(enc_prec_list) + 1):

        line_style_index = i % len(line_style_list)
        color_index = i % len(color_list)
        if i == 0:
            if (len(legend_str_list) > 0):
                plt.plot(qg_reca_list, qg_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index],
                         label=legend_str_list[0])
            else:
                plt.plot(qg_reca_list, qg_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index])
        else:
            bf_reca_list = enc_reca_list[i - 1]
            bf_prec_list = enc_prec_list[i - 1]
            if (len(legend_str_list) > 0):
                plt.plot(bf_reca_list, bf_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index],
                         label=legend_str_list[i])
            else:
                plt.plot(bf_reca_list, bf_prec_list, color=color_list[color_index],
                         lw=3, linestyle=line_style_list[line_style_index])

    # Set the axis minimum and maximum values
    #
    plt.xlim(xmin=x_min - margin, xmax=x_max + margin)
    plt.ylim(ymin=y_min - margin, ymax=y_max + margin)

    # Add the title(s) into the plot
    #
    if (title2 != None or title2 != ''):
        # plt.suptitle(title2, y=1.0, fontsize=title_font_size)
        plt.title(title1) #, fontsize=title_font_size)
    else:
        plt.title(title1) #, fontsize=title_font_size)

    plt.legend(loc='best') #, fontsize=axis_font_size)
    plt.show()
    # if (save_fig_name != None):
    #     plt.savefig(save_fig_name, bbox_inches='tight')
    #     plt.clf()
    #
    # else:
    #     input('Press return')
    #     plt.clf()


def calc_opt(attr_num_list, all_rec_list, q, padded, bf_len):
    num_attr_use = len(attr_num_list)

    # Get the average number of q-grams in the attribute values
    #
    total_num_q_gram = 0.0
    num_q_gram_list = []
    for i in range(num_attr_use):
        num_q_gram_list.append(0.0)
    total_num_val = 0

    # Calculate average number of q-grams of all attribute values
    #
    for rec_val_list in all_rec_list:
        attr_val_list = []
        for (i, attr_num) in enumerate(attr_num_list):
            attr_val = rec_val_list[attr_num]
            attr_val_list.append(attr_val)
            if (padded == True):
                attr_num_q_gram = len(attr_val) + q - 1
            else:
                attr_num_q_gram = len(attr_val) - q + 1
            num_q_gram_list[i] += attr_num_q_gram

        rec_val = ' '.join(attr_val_list)
        if (padded == True):
            total_num_q_gram += (len(rec_val) + q - 1)
        else:
            total_num_q_gram += (len(rec_val) - q + 1)

        total_num_val += 1

    attr_avrg_num_q_gram_list = []

    for i in range(num_attr_use):
        attr_avrg_num_q_gram_list.append(float(num_q_gram_list[i]) / total_num_val)

    avrg_num_q_gram = float(total_num_q_gram) / total_num_val
    num_hash_funct = int(round(math.log(2.0) * float(bf_len) /
                               avrg_num_q_gram))
    print('opt num_hash_funct =', num_hash_funct)
    print()
    return num_hash_funct
