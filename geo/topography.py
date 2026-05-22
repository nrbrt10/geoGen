from . import random, deque, np,  heapq
from scipy.spatial import Voronoi

def generate_voronoi_cells(width: int, height: int, n_seeds: int):
    points = np.array([
        [random.uniform(0, width), random.uniform(0, height)]
        for _ in range(n_seeds)
    ])

    vor = Voronoi(points)

    return points, vor

def classify_cells(adjacency, n_continents, valid_indices=None):
    n_cells = len(adjacency)
    continent_labels = [-1] * n_cells  # -1 = unassigned

    # Pick major continent seeds randomly
    if not valid_indices:
        continentSeeds = random.sample(range(n_cells), n_continents)
    else:
        continentSeeds = random.sample(valid_indices, n_continents)

    queue = deque()
    for cid, seed in enumerate(continentSeeds):
        continent_labels[seed] = cid
        queue.append(seed)

    while queue:
        current = queue.popleft()
        cid = continent_labels[current]

        for neighbor in adjacency[current]:
            if continent_labels[neighbor] == -1:
                continent_labels[neighbor] = cid
                queue.append(neighbor)
    
    return continent_labels

def assign_movement_vectors(continent_labels):
    unique_labels = set(continent_labels)
    vectors = {}
    for label in unique_labels:
        angle = random.uniform(0, 2*np.pi)
        magnitude = random.uniform(2, 5)  # arbitrary scale
        vectors[label] = np.array([np.cos(angle), np.sin(angle)]) * magnitude
    return vectors

def elevation_from_border(v_rel, max_convergent=50, min_divergent=-20):
    if v_rel > 0.1:
        return min(v_rel * 20, max_convergent)  # mountains
    elif v_rel < -0.1:
        return max(v_rel * 5, min_divergent)  # trenches
    else:
        return 0  # transform / neutral
    
def infer_elevation(points: np.array, vor: Voronoi, continent_labels, vectors):
    """
    For each Voronoi cell, assign an elevation based on neighboring continent interactions.
    Returns: array of elevation per cell
    """
    n_cells = len(points)
    centroids = { label: points[np.array(continent_labels) == label].mean(axis=0)
                 for label in set(continent_labels) }
    elevations = np.zeros(n_cells)

    for ridge_vertices, ridge_points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 in ridge_vertices:
            continue
        c1, c2 = continent_labels[ridge_points[0]], continent_labels[ridge_points[1]]
        if c1 == c2:
            continue  # internal ridge

        # compute unit vector along centroid connection
        vec = centroids[c2] - centroids[c1]
        vec /= np.linalg.norm(vec) + 1e-8
        # relative velocity along edge
        v_rel = np.dot(vectors[c1] - vectors[c2], vec)
        elev = elevation_from_border(v_rel)

        elevations[ridge_points[0]] += elev * (1 + random.uniform(1e-5, 1e-4))
        elevations[ridge_points[1]] += elev * (1 + random.uniform(1e-5, 1e-4))

    return elevations

def apply_sea_level(elevation, sea_percentile=50):
    """
    Converts elevation into land/sea by percentile.
    """
    sea_level = np.percentile(elevation, sea_percentile)

    land = {}
    for cell, h in enumerate(elevation):
        land[cell] = h - sea_level

    return land, sea_level

def classify_continents_v2(adjacency_graph, n_continents, noise=0.0):
    n_cells = len(adjacency_graph)
    continent_labels = np.full(n_cells, [-1], dtype=int)

    continent_seeds = random.sample(range(n_cells), n_continents)
    visit_order = []
    waves = {}
    
    for cid, seed in enumerate(continent_seeds):
        heapq.heappush(visit_order, (0, seed))
        waves[seed] = 0
        continent_labels[seed] = cid
    
    while visit_order:
        priority, current = heapq.heappop(visit_order)
        cid = continent_labels[current]
        wave = waves[current]

        for neighbor in adjacency_graph[current]:
            if continent_labels[neighbor] != -1:
                continue
            
            continent_labels[neighbor] = cid
            waves[neighbor] = wave + 1
            heapq.heappush(visit_order, (waves[neighbor] + random.uniform(0, noise), neighbor))

    return continent_labels

def compute_dependencies(elevated_cells_by_landmass, adjacency_by_landmass):
    waves = {}
    dependencies = {}
    queue = deque()

    for label, cells in elevated_cells_by_landmass.items():
        queue.clear()
        
        for cell in cells:
            waves[cell] = 0
            queue.append(cell)

        while queue:
            current = queue.popleft()
            dependencies[current] = []

            for neighbor in adjacency_by_landmass[current]:
                if neighbor not in waves:
                    waves[neighbor] = waves[current] + 1
                    queue.append(neighbor)
                elif waves[neighbor] == waves[current]:
                    continue
                elif waves[neighbor] == waves[current] - 1:
                    dependencies[current].append(neighbor)

    return dependencies, waves

def propagate_elevations(elevations, continent_labels, adjacency_by_landmass, decay=0.25, mode=None):
    new_elevations = elevations.copy()
    elevated_cells_by_landmass = {label: [idx for idx, tag in enumerate(continent_labels) if tag == label and elevations[idx] != 0] for label in np.unique(continent_labels)}
    dependencies, wave = compute_dependencies(elevated_cells_by_landmass, adjacency_by_landmass)

    for idx, sources in dependencies.items():
        if not sources:
            continue

        n = len(sources)
        avg = sum(new_elevations[sources]) / n
        new_elevations[idx] = avg * decay ** (wave[idx] / n ** decay) * (1 + random.uniform(1e-5, 1e-4))

    if mode == 'debug':
        return new_elevations, dependencies, wave
    
    return new_elevations

def scale_elevations(elevations, sea_level, percentile=95, percentile_map=8800):
    zero_sea_level_elevations = elevations + (-1 * sea_level)
    k = percentile_map / np.percentile(elevations, [percentile])
    elevations_m = zero_sea_level_elevations * k
    return elevations_m