# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

from __future__ import division

import numpy as np
import string
import copy
import sys
import itertools

from scipy.io import wavfile
from operator import itemgetter
from scipy.signal import butter, lfilter, square
from sklearn.decomposition import PCA


Codebook = {
    '.-':       'A', '-...':    'B', '-.-.':    'C', '-..':     'D', '.':       'E',
    '..-.':     'F', '--.':     'G', '....':    'H', '..':      'I', '.---':    'J',
    '-.-':      'K', '.-..':    'L', '--':      'M', '-.':      'N', '---':     'O',
    '.--.':     'P', '--.-':    'Q', '.-.':     'R', '...':     'S', '-':       'T',
    '..-':      'U', '...-':    'V', '.--':     'W', '-..-':    'X', '-.--':    'Y',
    '--..':     'Z', '.----':   '1', '..---':   '2', '...--':   '3',
    '....-':    '4', '.....':   '5', '-....':   '6', '--...':   '7',
    '---..':    '8', '----.':   '9', '-----':   '0'
}

possible_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class Morse(object):

    def __init__(self):
        self.data = []
        self.true_length = 0
        self.dit_length = None
        self.sample_rate = None
        self.text = ""
        self.morse_code = ""
        self.total_dits = 0
        self.audio_data = None

    def read_wav_file(self, filename):
        (self.sample_rate, self.audio_data) = wavfile.read(filename)

    def get_morse_code_from_text(self, text=None):
        if text is None:
            text = self.text
        return string.join([ReverseCodebook[x] if x in ReverseCodebook else "?" for x in text], ' ')

    def get_text_from_code(self, code=None):
        if code is None:
            code = self.code
        return string.join([Codebook[x] if x in Codebook else "?" for x in string.split(code,' ')], '')

    def get_binary_from_code(self, code=None):
        if code is None:
            code = self.morse_code
        code = code.replace(".", "10")
        code = code.replace("-", "1110")
        code = code.replace(" ", "00")
        #return [int(x) for x in list("0" * 7 + code + "0" * 16)]
        return [int(x) for x in code]

    def get_code_from_binary_array(self, binary=""):

        b = string.join([str(x) for x in binary], '')

        res = ""
        while len(b) > 0:
            if b[0:2] == "10":
                res += "."
                b = b[2:]
                continue
            if b[0:4] == "1110":
                res += "-"
                b = b[4:]
                continue
            if b[0:2] == "00":
                res += " "
                b = b[2:]
                continue
            res += "x"
            b = b[2:]

        return res

    def estimate_total_dits(self, data=None):
        if data is None:
            data = butter_bandpass_filter(self.audio_data, int(600 - 50 / 2), int(600 + 50 / 2), self.sample_rate, order=1)
        result = []
        t = np.linspace(0, len(data) / self.sample_rate, len(data), endpoint=False)
        for num_dits in xrange(220, 360):
            dits = (-square(np.pi * num_dits / (len(data) / self.sample_rate) * t) + 1) / 2
            err = abs(dits - (abs(data) / np.max(data)))
            result.append((num_dits, np.mean(err)))

        return sorted(result, key=itemgetter(1))[0][0]

    def get_legal_8gram_solutions(self, binary):

        legal_8grams = ['00101000', '10111011', '00111010', '00111011', '00100011', '10001011', '10001010', '10001110', '11101000', '10101110', '10111000', '00100010', '11101110', '10101000', '00101011', '00101010', '00111000', '00101110', '10001000', '10111010', '11101011', '11101010', '11100010', '11100011', '10100011', '10100010', '10101010', '10101011']

        b = copy.copy(binary)

        ## pad data to multiples of 8
        if (len(b) % 4) != 0:
            b = b + [0,0]
        if (len(b) % 8) != 0:
            b = b + [0,0,0,0]

        bp = np.array([b[int(i):(i + 8)] for i in range(0, len(b) - 4, 4)])

        possible_moves = {}

        binary_repaired = b

        for pos,eightgram in enumerate(bp):
            j = string.join([str(v) for v in eightgram], '')
            if j not in legal_8grams:
                #print >> sys.stderr, "found illegal 8gram %s at position %d" % (j, pos)

                best_fix = []
                for k in legal_8grams:
                    moves = 8 - [m1 == m2 for (m1, m2) in zip(j,k)].count(True)
                    best_fix.append([moves,k])
                lowest_moves = sorted(best_fix)[0][0]
                for moves,k in best_fix:
                    if moves == lowest_moves:
                        #print >> sys.stderr, "%s%s in %d moves" % (" " * pos * 4, k, moves)

                        if pos * 4 not in possible_moves:
                            possible_moves[pos * 4] = []
                        if pos * 4 + 4 not in possible_moves:
                            possible_moves[pos * 4 + 4] = []

                        idx = pos * 4
                        possible_moves[idx].append(k[:4])
                        idx = pos * 4 + 4
                        possible_moves[idx].append(k[4:])

        to_be_deleted = []
        for p in possible_moves:
            possible_moves[p] = list(set(possible_moves[p]))
            if len(possible_moves[p]) == 1:
                binary_repaired[p:p + 4] = possible_moves[p][0]
                to_be_deleted.append(p)

        for p in to_be_deleted:
            del possible_moves[p]

        #print >> sys.stderr, "REPAIRED:",string.join([str(v) for v in binary_repaired],'')
        #print >> sys.stderr, possible_moves

        positions = possible_moves.keys()
        moves = possible_moves.values()

        sort_this = []

        for move in itertools.product(*moves):
            binary_repaired_copy = copy.copy(binary_repaired)

            for p,mv in zip(positions,move):
                binary_repaired_copy[p:p + 4] = mv

            binary_repaired_copy = [int(c) for c in binary_repaired_copy][:len(binary)]
            #print >> sys.stderr, binary_repaired_copy

            text = self.get_text_from_code(self.get_code_from_binary_array(binary_repaired_copy))
            sort_this.append([len(text), text.count("?"), binary_repaired_copy])
            #print >> sys.stderr, text, len(text), text.count("?")

        return [v[2] for v in sorted(sort_this, key=itemgetter(1))]

    def get_fft_values(self):

        x = np.array([self.audio_data[int(i * self.dit_length):(i + 1) * self.dit_length] for i in range(0, int(self.total_dits))])

        this_fft = []
        for z in x:
            f = np.abs(np.fft.fft(z))
            f = f[:int(self.dit_length / 2)]
            this_fft.append(f.tolist())

        pca = PCA(n_components=1)
        this_fft_pca = pca.fit_transform(np.array(this_fft))

        return this_fft_pca

ReverseCodebook = {}
for x in Codebook:
    ReverseCodebook[Codebook[x]] = x
