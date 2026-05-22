from . import np, helpers, math

def infer_temperatures(
        points: np.array,
        elevations: np.array,
        sea_level: float,
        equator_y,
        equator_baseline=35,
        pole_baseline=-38,
        dist_exp=2,
        elevation_exp=2,
        elevation_effect_strength=.33):
    temps = np.zeros(len(points))
    normalized_temps = np.zeros(len(points))
    elev_max = elevations[elevations >= sea_level].max()

    for idx, point in enumerate(points):
        norm_dist = abs(point[1] - equator_y) / equator_y
        dist_factor = helpers.clamp(1 - (norm_dist ** dist_exp + 1e-10), 0, 1)
        if elevations[idx] > sea_level: 
            norm_elev = abs(elevations[idx] - sea_level) / (elev_max - sea_level)
            elev_effect = norm_elev ** elevation_exp * elevation_effect_strength
        else:
            elev_effect = 0
        temps[idx] = pole_baseline + (equator_baseline - pole_baseline) * (dist_factor - elev_effect)
        normalized_temps[idx] = helpers.clamp(dist_factor - elev_effect, 0, 1)

    return normalized_temps, temps

def infer_moisture(elevations, normalized_temperatures, sea_level):
    moisture = np.full(len(elevations), [-1], dtype=np.float32)
    for idx, temp in enumerate(normalized_temperatures):
        moisture[idx] = temp * (1 if elevations[idx] <= sea_level else .25)

    return moisture

def infer_precipitation(
        points,
        normalized_temperatures,
        base_moisture,
        adjacency_graph,
        wind_vectors,
        tangent_vectors=None,
        diffuse=True,
        diffuse_alpha=0.1,
        steps=40, precipitation_fraction=.33, rising_fraction=0.175, evaporation_rate=.05, capacity_scale=.85, capacity_floor=1e-5, transport_rate=.33, p=1.5):
    transport = base_moisture.copy()
    saturated = np.zeros(len(transport), dtype=np.float32)
    precipitation = np.zeros(len(points), dtype=np.float32)


    for step in range(steps):
        transport_delta = np.zeros(len(transport), dtype=np.float32)
        saturated_delta = np.zeros(len(transport), dtype=np.float32)
        for idx, m in enumerate(transport):
            c1 = points[idx]
            elegible_neighbors = []
            alignment_sum = 0
            for neighbor in adjacency_graph[idx]:
                c2 = points[neighbor]
                v = c2 - c1 if tangent_vectors is None else tangent_vectors[idx][neighbor]
                d_c1_c2 = np.dot(wind_vectors[idx], v)
                if d_c1_c2 > 0:
                    alignment_sum += d_c1_c2
                    elegible_neighbors.append((neighbor, d_c1_c2))

            if len(elegible_neighbors) == 0:
                continue

            for neighbor, dot in elegible_neighbors:
                alignment_weight = dot / alignment_sum
                m_flux = m * transport_rate * alignment_weight
                s_flux = saturated[idx] * transport_rate * alignment_weight
                transport_delta[idx] -= m_flux
                transport_delta[neighbor] += m_flux
                saturated_delta[idx] -= s_flux
                saturated_delta[neighbor] += s_flux

        transport += transport_delta
        saturated += saturated_delta

        if diffuse:
            transport = math.diffuse_field(transport, adjacency_graph, alpha=diffuse_alpha, iterations=1)
            saturated = math.diffuse_field(saturated, adjacency_graph, alpha=diffuse_alpha, iterations=1)

        uplifted_moisture = np.power(transport, p) * rising_fraction * normalized_temperatures
        transport -= uplifted_moisture
        surface_capacity = np.maximum(capacity_scale * normalized_temperatures ** 2, capacity_floor)
        excess = np.maximum(0, transport - surface_capacity)
        precipitation += (excess * precipitation_fraction) + (saturated * precipitation_fraction) + uplifted_moisture
        saturated += excess * (1 - precipitation_fraction) - saturated * precipitation_fraction
        transport -= excess
        rh = np.minimum(math.array_safe_divide(transport, surface_capacity, 0), 1)
        evaporation = np.maximum(transport * evaporation_rate * (1 - rh), 0)
        transport += evaporation
    
    precipitation += saturated
        
    return precipitation, transport, evaporation

def scale_precipitaion(precipitation, percentile=95, percentile_map=2500):
    k = percentile_map / np.percentile(precipitation, [percentile])
    precipitation_mm = precipitation * k
    return precipitation_mm