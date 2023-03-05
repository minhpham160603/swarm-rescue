import numpy as np

def crop(map):
    return map, 0, 0
    # print(np.nonzero(map), map.shape)
    row, col = np.nonzero(np.atleast_1d(map))
    # print('asdf', row, col, type(row), col.shape)
    if row.any() == False:
        row = [0, map.shape[0]-1]
        col = [0, map.shape[1]-1]
    # print('asdf', row, col, type(row))
    return map[min(row):max(row)+1][min(col):max(col)+1], min(row), min(col)

def enlarge(map):
    map, a, b = crop(map)
    row, col = map.shape
    row_add = row//4
    col_add = col//4
    large_map = np.zeros((2*row_add+row, 2*col_add+col), dtype="int")
    large_map[row_add:-row_add, col_add:-col_add] = map
    return large_map, a+row_add, b+col_add