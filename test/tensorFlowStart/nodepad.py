# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np

full = np.full((2), 1, float)
mat = np.array([[1, 2], [3, 4]], float)

print full
print
print mat
print
print np.dot(full, mat) + 4
