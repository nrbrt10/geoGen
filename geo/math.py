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

def normalize_array(x: np.array, min: float, max: float):
    return (x - min) / (max - min)

def project_3d_to_2d(x, y, z, type='equirectangular') -> tuple[np.array, np.array]:
    '''
    Takes transposed 3d vector and converts to degrees
    '''
    lon = np.arctan2(y, x)
    if type == 'equirectangular':
        lat = np.arcsin(z)
    elif type == 'lambert':
        lat = z
    radians = np.column_stack([lon, lat])
    degrees = np.degrees(radians)
    return radians, degrees

def chord_to_tangent(tangent_point, v: np.array):
    normalized = tangent_point / np.linalg.norm(tangent_point)
    v_tan = v - (v @ normalized)[:, np.newaxis] * normalized
    return v_tan

def build_tangent_vectors(points, adjacency):
    vectors = [-1] * len(points)
    for idx, point in enumerate(points):
        v = points[np.array(list(adjacency[idx]))] - point
        v_tan = chord_to_tangent(point, v)
        vectors[idx] = {neighbor: tan for neighbor, tan in zip(adjacency[idx], v_tan)}
    return vectors

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

def compute_basis_vectors(coords_radians):
    lon = coords_radians[:,0:1]
    lat = coords_radians[:,1:2]

    e_lon = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros(len(lon))])
    e_lat = np.column_stack([-np.sin(lat) * np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])

    return e_lon, e_lat

def rebuild_3d_vectors(vectors_2d, e_lon, e_lat):
    v_lon = vectors_2d[:,0:1]
    v_lat = vectors_2d[:,1:2]

    vectors_3d = v_lon * e_lon + v_lat * e_lat
    return vectors_3d

def geodesic_distance(a, b):
    dot = np.clip(np.dot(a, b), -1, 1)
    return np.arccos(dot)

def random_tangent_vector(p, c, seed=123):
    np.random.seed(seed + c)
    r = np.random.randn(3)
    r = r - np.dot(r, p) * p
    return r / np.linalg.norm(r)