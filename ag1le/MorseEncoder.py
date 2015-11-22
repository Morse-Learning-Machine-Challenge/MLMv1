#!/usr/bin/env python
# MorseEncoder.py  - Morse Encoder to generate training material for neural networks
# Generates raw signal waveforms with Gaussian noise and QSB (signal fading) effects
# Provides also the training target variables in separate columns. Example usage:
#
# WPM= 40 # speed 40 words per minute
# Tq = 4. # QSB cycle time in seconds (typically 5..10 secs)
# sigma = 0.02 # add some Gaussian noise
# P = signal('QUICK BROWN FOX JUMPED OVER THE LAZY FOX ',WPM,Tq,sigma)
# from matplotlib.pyplot import  plot,show,figure,legend
# from numpy.random import normal
# figure(figsize=(12,3))
# lb1,=plot(P.t,P.sig,'b',label="sig")
# lb2,=plot(P.t,P.dit,'g',label="dit")
# lb3,=plot(P.t,P.dah,'g',label="dah")
# lb4,=plot(P.t,P.ele,'m',label="ele")
# lb5,=plot(P.t,P.chr,'c',label="chr")
# lb6,=plot(P.t,P.wrd,'r*',label="wrd")
# legend([lb1,lb2,lb3,lb4,lb5,lb6])
# show()
# P.to_csv("MorseTest.csv")
#
# Copyright (C) 2015   Mauri Niininen, AG1LE
#
#
# MorseEncoder.py is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MorseEncoder.py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with bmorse.py.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from numpy import sin,pi
from numpy.random import normal
pd.options.mode.chained_assignment = None  #to prevent warning messages

Morsecode = {
 '!': '-.-.--',
 '$': '...-..-',
 "'": '.----.',
 '(': '-.--.',
 ')': '-.--.-',
 ',': '--..--',
 '-': '-....-',
 '.': '.-.-.-',
 '/': '-..-.',
 '0': '-----',
 '1': '.----',
 '2': '..---',
 '3': '...--',
 '4': '....-',
 '5': '.....',
 '6': '-....',
 '7': '--...',
 '8': '---..',
 '9': '----.',
 ':': '---...',
 ';': '-.-.-.',
 '<AR>': '.-.-.',
 '<AS>': '.-...',
 '<HM>': '....--',
 '<INT>': '..-.-',
 '<SK>': '...-.-',
 '<VE>': '...-.',
 '=': '-...-',
 '?': '..--..',
 '@': '.--.-.',
 'A': '.-',
 'B': '-...',
 'C': '-.-.',
 'D': '-..',
 'E': '.',
 'F': '..-.',
 'G': '--.',
 'H': '....',
 'I': '..',
 'J': '.---',
 'K': '-.-',
 'L': '.-..',
 'M': '--',
 'N': '-.',
 'O': '---',
 'P': '.--.',
 'Q': '--.-',
 'R': '.-.',
 'S': '...',
 'T': '-',
 'U': '..-',
 'V': '...-',
 'W': '.--',
 'X': '-..-',
 'Y': '-.--',
 'Z': '--..',
 '\\': '.-..-.',
 '_': '..--.-',
 '~': '.-.-'}
    

def encode_morse(cws):
    s=[]
    for chr in cws:
        try: # try to find CW sequence from Codebook
            s += Morsecode[chr]
            s += ' '
        except:
            if chr == ' ':
                s += '_'
                continue
            print "error: '%s' not in Codebook" % chr
    return ''.join(s)



def len_dits(cws):
    # length of string in dit units, include spaces
    val = 0
    for ch in cws:
        if ch == '.': # dit len + el space 
            val += 2
        if ch == '-': # dah len + el space
            val += 4
        if ch==' ':   #  el space
            val += 2
        if ch=='_':   #  el space
            val += 7
    return val


def signal(cw_str,WPM,Tq,sigma):
    # for given CW string i.e. 'ABC ' 
    # return a pandas dataframe with signals and  symbol probabilities
    # WPM = Morse speed in Words Per Minute (typically 5...50)
    # Tq  = QSB cycle time (typically 3...10 seconds) 
    # sigma = adds gaussian noise with standard deviation of sigma to signal
    cws = encode_morse(cw_str)
    #print cws
    # calculate how many milliseconds this string will take at speed WPM
    ditlen = 1200/WPM # dit length in msec, given WPM
    msec = ditlen*(len_dits(cws)+7)  # reserve +7 for the last pause
    t = np.arange(msec)/ 1000.       # time array in seconds
    ix = range(0,msec)               # index for arrays

    # Create a DataFrame and initialize
    col =["t","sig","dit","dah","ele","chr","wrd"]
    P = pd.DataFrame(index=ix,columns=col)
    P.t = t              # keep time  
    P.sig=np.zeros(msec) # signal stored here
    P.dit=np.zeros(msec) # probability of 'dit' stored here
    P.dah=np.zeros(msec) # probability of 'dah' stored here
    P.ele=np.zeros(msec) # probability of 'element space' stored here
    P.chr=np.zeros(msec) # probability of 'character space' stored here
    P.wrd=np.zeros(msec) # probability of 'word space' stored here
    P.spd=np.ones(msec)*WPM #speed stored here 

    
    #pre-made arrays with multiple(s) of ditlen
    z = np.zeros(ditlen) 
    z2 = np.zeros(2*ditlen)
    z4 = np.zeros(4*ditlen)
    dit = np.ones(ditlen)
    dah = np.ones(3*ditlen)
      
    # For all dits/dahs in CW string generate the signal, update symbol probabilities
    i = 0
    for ch in cws:
        if ch == '.':
            dur = len(dit)
            P.sig[i:i+dur] = dit
            P.dit[i:i+dur] = dit
            i += dur
            dur=len(z)
            P.sig[i:i+dur] = z
            P.ele[i:i+dur] = np.ones(dur)
            i += dur

        if ch == '-':
            dur = len(dah)
            P.sig[i:i+dur] = dah
            P.dah[i:i+dur]=  dah
            i += dur            
            dur=len(z)
            P.sig[i:i+dur] = z
            P.ele[i:i+dur] = np.ones(dur)
            i += dur

        if ch == ' ':
            dur = len(z2)
            P.sig[i:i+dur] = z2
            P.chr[i:i+dur]=  np.ones(dur)
            i += dur
        if ch == '_':
            dur = len(z4)
            P.sig[i:i+dur] = z4
            P.wrd[i:i+dur]=  np.ones(dur)
            i += dur
    if Tq > 0.:  # QSB cycle time impacts signal amplitude
        qsb = 0.5 * sin((1./float(Tq))*t*2*pi) +0.55
        P.sig = qsb*P.sig
    if sigma >0.:
        P.sig += normal(0,sigma,len(P.sig))
    return P
