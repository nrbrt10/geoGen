from . import np

def compute_gaussian_terms(means, std_devs, values_vector):
    return np.power((means - values_vector) / std_devs, 2)

def gaussian_classificator(means, std_devs, values_vector, mode=None):
    expressions = np.power((means - values_vector) / std_devs, 2)
    scores = np.sum(expressions, axis=1)
    m = scores.min()
    if mode == 'debug':
        return scores, m

    return np.where(scores == m)

def array_safe_divide(n: np.array, d: np.array, fallback=0):
    return np.where(d != 0, np.divide(n, d, where=d != 0), fallback)

def diffuse_field(scalar_field, adjacency_graph, alpha=0.2, iterations=5):
    current = scalar_field.copy()
    
    for i in range(iterations):
        delta = np.zeros(len(current))
        for idx in range(len(current)):
            neighbors = list(adjacency_graph[idx])
            all_cells = [idx] + neighbors
            local_mean = current[all_cells].mean()
            delta[idx] = local_mean - current[idx]
        
        current += delta * alpha
    
    return current

def compute_voronoi_areas_r1(points, regions, vertices):
    '''
    For unit sphere (r=1)
    '''
    areas = np.zeros(len(points))
    for idx, region in enumerate(regions):
        a = points[idx]
        for i, vertex in enumerate(region):
            b = vertices[vertex]
            c = vertices[region[(i+1) % len(region)]]
            cross = np.cross(b - a, c - a)
            areas[idx] += 0.5 * np.linalg.norm(cross)
    return areas