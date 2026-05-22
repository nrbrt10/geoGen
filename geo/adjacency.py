def build_adjacency(discrete_units, ridge_points):
    adjacency = {i: set() for i in range(len(discrete_units))}
    for a, b, in ridge_points:
        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency

def build_landmass_adjacency(discrete_units, ridge_points, continent_labels):
    adjacency = {i: set() for i in range(len(discrete_units))}
    for a, b, in ridge_points:
        if continent_labels[a] == continent_labels[b]:
            adjacency[a].add(b)
            adjacency[b].add(a)

    return adjacency

def build_land_adjancency_graph(points, ridge_points, elevations, sea_level):
    adjacency = {i: set() for i in range(len(points))}
    for a, b, in ridge_points:
        if elevations[a] > sea_level and elevations[b] > sea_level:
            adjacency[a].add(b)
            adjacency[b].add(a)

    return adjacency