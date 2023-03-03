import numpy as np

def crop(map):
    row, col = np.nonzero(map)
    map = map[min(row):max(row)][min(col):max(col)]
    return min(row), min(col)

def enlarge(map):
    a, b = crop(map)
    row, col = map.shape
    row_add = row//4
    col_add = col//4
    large_map = np.zeros((2*row_add+row, 2*col_add+col), dtype="int")
    large_map[row_add:-row_add][col_add:-col_add] = map
    map = large_map
    return a+row_add, b+col_add