from . import np

def find_root(parents: np.array, u: int):
    if parents[u] == u:
        return u
    visited = []
    current = u
    while parents[current] != current:
            visited.append(current)
            current = parents[current]
    for i in visited:
        parents[i] = current
            
    return current

def get_members(id, array: np.array):
    return np.where(array == id)[0]