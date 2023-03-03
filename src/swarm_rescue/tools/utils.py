import numpy as np

def crop(map):
    # print(np.nonzero(map), map.shape)
    row, col = np.nonzero(np.atleast_1d(map))
    map = map[min(row):max(row)][min(col):max(col)]
    return min(row), min(col)

def enlarge(map):
    a, b = crop(map)
    row, col = map.shape
    row_add = row//4
    col_add = col//4
    large_map = np.zeros((2*row_add+row, 2*col_add+col), dtype="int")
    # print(map.shape, row_add, col_add)
    large_map[row_add:-row_add, col_add:-col_add] = map
    return large_map, a+row_add, b+col_add