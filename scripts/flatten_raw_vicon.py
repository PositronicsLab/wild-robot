import wb_raw_vicon_reader as reader
import numpy as np
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

a = reader.read(infile, False)
np.savetxt(outfile, a, delimiter=' ', fmt='%1.5e')
