from __future__ import division

import morse
import numpy as np
import string
import glob
import sys
import itertools
import copy

from operator import itemgetter
from sklearn.cluster import KMeans
from ffnet import ffnet, mlgraph, savenet, loadnet

TRAIN_NEURAL_NET = False

m = morse.Morse()

def repair_binary(binary, certainty, max_moves):

    solutions = []

    top_uncertainty_positions = [v[0] for v in sorted(zip(range(len(binary)), certainty), key=itemgetter(1))]

    for num_moves in range(0,max_moves + 1):

        if num_moves == 1:
            ## try all positions if only doing one move
            positions = top_uncertainty_positions
        else:
            ## try 100 most uncertain positions if doing more than one move
            positions = top_uncertainty_positions[:100]

        p = itertools.combinations(positions, num_moves)
        for mov in p:

            sum_certainty = 0

            b = copy.copy(binary)
            for flip_pos in mov:
                b[flip_pos] = 1 - b[flip_pos]
                sum_certainty += certainty[flip_pos]

            code = m.get_code_from_binary_array(b)
            t = m.get_text_from_code(code)

            if len(t) == 20 and "?" not in t:
                solutions.append([t, float(sum_certainty)])

        if len(solutions) > 0:
            return solutions

    return []

def get_nnet_input_data(m):

    m.total_dits = m.estimate_total_dits()
    m.dit_length = len(m.audio_data) / m.total_dits

    fft_values = m.get_fft_values()

    cluster = KMeans(init='k-means++', n_clusters=2, n_init=20, precompute_distances=True)
    cluster.fit(fft_values)

    ## ==================================
    ## compute distance to cluster center
    ## ==================================
    center = np.mean(cluster.cluster_centers_)
    distances = abs(fft_values - center)
    certainty = distances / max(distances)
    certainty = certainty[7:-16]

    proba = ((fft_values - center) / max(distances) / 2) + 0.5

    ## invert if necessary - KMeans cluster assignment is unstable
    if np.mean(proba[:7]) > 0.5:
        proba = 1 - proba

    proba = [0.0,0.0,0.0,0.0] + proba[7:-16].reshape(-1).tolist() + [0.0,0.0,0.0,0.0]

    input_data = np.array([proba[i:i + 9] for i in range(0, len(proba) - 8)]).astype(np.float32)

    return input_data, certainty

## ===============================
## TRAIN NEURAL NET
## ===============================

conec = mlgraph((9, 9, 1))
net = ffnet(conec)

if TRAIN_NEURAL_NET:

    # training files for neural net were generated using trainingfiles/generate_morse.m
    # list of solution texts is included in morse_text.txt

    X_train = None
    Y_train = None

    for filename in sorted(glob.glob("trainingfiles/*.wav")):
        (f, snr, wpm, solution) = string.split(filename, "_")

        print >> sys.stderr, "generating neural net input data for file %s" % f

        m.text = solution[:-4]

        ## get binary morse representation from solution text
        binary_solution = m.get_binary_from_code(m.get_morse_code_from_text())

        m.read_wav_file("%s" % filename)

        input_data, certainty = get_nnet_input_data(m)
        output_data = np.array(binary_solution, dtype=np.float32)

        if X_train is None:
            X_train = input_data
            Y_train = output_data
        else:
            X_train = np.concatenate((X_train, input_data), axis=0)
            Y_train = np.concatenate((Y_train, output_data), axis=0)

    net.train_tnc(X_train, Y_train, nproc=4, maxfun=200000, messages=2)
    savenet(net, "morse.net")

else:
    net = loadnet("morse.net")

## ===============================
## GENERATE SUBMISSION
## ===============================
with open('sampleSubmission.csv') as f:
    trainingset = f.read().split("\n")

files = [(('000' + x.split(",")[0])[-3:], x.split(",")[1]) for x in trainingset[1:] if "," in x]

f = open('submission.csv','w')
f.write("ID,Prediction\n")

for filenum, m.text in files:

    if len(m.text) > 0:
        f.write("%s,%s\n" % (filenum, m.text))
        continue

    print >> sys.stderr, 'FILE', filenum

    m.read_wav_file("cw%s.wav" % filenum)

    input_data, certainty = get_nnet_input_data(m)

    net_output = np.array(net.call(input_data)).reshape(-1).tolist()
    binary_net_output = [1 if v > 0.5 else 0 for v in net_output]

    ## =============
    ## repair binary
    ## =============
    for max_moves in range(1,6):
        repaired_texts = []
        for possible_solution in m.get_legal_8gram_solutions(binary_net_output):
            print >> sys.stderr, "searching best solution for %s with up to %d move(s)" % (m.get_text_from_code(m.get_code_from_binary_array(possible_solution)), max_moves)
            repaired_texts.extend(repair_binary(possible_solution, certainty, max_moves=max_moves))

        if len(repaired_texts) > 0:
            break

    if len(repaired_texts) == 0:
        print >> sys.stderr, "ERROR: no valid solution found"
        sys.exit()

    ## sort repaired solutions by minimum total distance to cluster center
    best_repaired_text = sorted(repaired_texts, key=itemgetter(1))[0][0]

    print >> sys.stderr, "best solution: %s" % best_repaired_text
    f.write("%s,%s\n" % (filenum, best_repaired_text))

f.close()
