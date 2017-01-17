import numpy as np
import sys

from operator import itemgetter

symbols = sys.argv[1]

All = np.genfromtxt(symbols,delimiter=';',dtype=None)
All = All[1:(len(All) - 1)]
args = All[:,2]
args = map(int, args)
store_index = [(index, training) for index, training in enumerate(args)]
sorted_index = sorted(store_index, key=itemgetter(1), reverse=True)
for i,t in sorted_index[0:25]:
	print int(All[i,0])

"""
i = 0;
with open('test-data.csv') as f:
    for line in f:
        if (i == 0):
        	continue:
        else:
        	i = 1
"""
