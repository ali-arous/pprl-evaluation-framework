# blocking.py - Module that contains core classes for the evaluation framework
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

from framework import PreEncodingBlockingMethod, PostEncodingBlockingMethod
from linkageUtils import hlsh_blocking, minhash_blocking, init_minhash


class HLSHBlocking(PostEncodingBlockingMethod):
    def __init__(self, block_hlsh_num_seg):
        self.block_hlsh_num_seg = block_hlsh_num_seg
        self.MAX_BLOCK_SIZE = 100

    def set_max_block_size(self, size):
        self.MAX_BLOCK_SIZE = size

    def generate_blocks(self, rec_bf_dict):
        return hlsh_blocking(rec_bf_dict, self.block_hlsh_num_seg, self.MAX_BLOCK_SIZE)


class MinHashBlocking(PreEncodingBlockingMethod):
    def __init__(self, lsh_band_size, lsh_num_band):
        self.lsh_band_size = lsh_band_size
        self.lsh_num_band = lsh_num_band
        self.MAX_BLOCK_SIZE = 100

    def set_max_block_size(self, size):
        self.MAX_BLOCK_SIZE = size

    def generate_blocks(self, rec_q_gram_dict):
        coeff_a_list, coeff_b_list = init_minhash(self.lsh_band_size, self.lsh_num_band)
        return minhash_blocking(rec_q_gram_dict, coeff_a_list,
                         coeff_b_list, self.lsh_band_size,
                         self.lsh_num_band, self.MAX_BLOCK_SIZE)
