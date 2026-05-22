from . import np, deque, heapq, math
from .helpers import clamp

def generate_drainage_data(elevations: np.array, adjacency: dict, sea_level: float, base_absorption=.4):
    directionality = np.full(len(elevations), [-3], dtype=np.int32)
    absorption = np.full(len(elevations), [-1], dtype=np.float32)
    slopes = np.full(len(elevations), [-1], dtype=np.float32)

    DRAIN_INLAND = np.int32(-1)
    DRAIN_TO_SEA = np.int32(-2)

    for idx, elevation in enumerate(elevations):
        if elevation <= sea_level:
            continue

        lowest = None
        for neighbor in adjacency[idx]:
            if elevations[neighbor] < elevation:
                if lowest is None:
                    lowest = neighbor
                elif elevations[neighbor] < elevations[lowest]:
                    lowest = neighbor
                    
        if lowest is None:
            lowest = DRAIN_INLAND
        elif elevations[lowest] <= sea_level:
            directionality[lowest] = DRAIN_TO_SEA
                
        directionality[idx] = lowest
        if lowest != -1:
            slopes[idx] = elevation - elevations[lowest]
        else:
            slopes[idx] = 1
        absorption[idx] = base_absorption * clamp(1/slopes[idx], 0, 1) ** 1.75
        

    return directionality, absorption, slopes

def invert_drainage_array(drainage_array: np.array):
    inv_drainage = {}
    for idx, pointer in enumerate(drainage_array):
        if pointer == -3:
            continue
        
        upstream_idx = np.where(drainage_array == idx)[0].tolist()
        if len(upstream_idx) > 0:
            inv_drainage[idx] = upstream_idx
            
    return inv_drainage

def label_watersheds(drainage_array: np.array):
    sinks = [np.int32(idx) for idx, pointer in enumerate(drainage_array) if pointer in [-1, -2]]
    inverted_drainage_array = invert_drainage_array(drainage_array)
    watersheds_array = np.full(len(drainage_array), [-1], dtype=np.int32)
    watersheds = {}
    for sink in sinks:
        stack = [sink]
        watersheds[np.int32(sink)] = set()
        while stack:
            current = stack.pop()
            if watersheds_array[current] == sink:
                continue
            
            if drainage_array[current] != -2:
                watersheds_array[current] = sink

            if current in inverted_drainage_array:
                stack.extend(inverted_drainage_array[current])
    
    return watersheds_array

def drainage_dependencies(inverted_drainage_graph: dict) -> dict:
    dependencies_graph = {}
    for idx, dependencies in inverted_drainage_graph.items():
        dependencies_graph[idx] = len(dependencies)

    return dependencies_graph

def compute_flow_volume(drainage_array: np.array, rainfall: np.array, absorption: np.array, slopes: np.array, soil_capacity: np.array, evaporation_rate: np.array | float=.175, soil_permeability=.5, drainage_efficiency=1.2):
    sources = [idx for idx, val in enumerate(drainage_array) if idx not in drainage_array and val not in [-2, -3]]
    dependency_count = drainage_dependencies(invert_drainage_array(drainage_array))
    inflows = np.zeros(shape=len(drainage_array))
    saturation = np.full(len(drainage_array), [-1], dtype=np.float32)
    drainage_volumes = np.zeros(shape=len(drainage_array))

    queue = deque()
    queue.extend(sources)

    if isinstance(evaporation_rate, float):
        evaporation_rate = np.full(len(drainage_array), evaporation_rate)

    while queue:
        current = queue.popleft()

        if drainage_array[current] in [-2, -3]:
            continue

        if slopes[current] == -1 or absorption[current] == -1:
            raise ValueError(f'Slopes or absorption data invalid at: {current}')

        total_load = rainfall[current] + inflows[current]
        saturation[current] = np.clip(math.array_safe_divide(total_load * absorption[current] * (1 - evaporation_rate[current]), soil_capacity[current], 0), 0, 2)
        runoff = total_load * (1 - evaporation_rate[current]) * (1 - absorption[current])
        drainage_volumes[current] = runoff

        if drainage_array[current] in [-1]:
            continue

        inflows[drainage_array[current]] += runoff
        dependency_count[drainage_array[current]] -= 1

        if dependency_count[drainage_array[current]] == 0:
            queue.append(drainage_array[current])

    return drainage_volumes, saturation

def build_drainage_segments_sv(drainage_array: np.array, regions_to_ridge_points: dict):
    end_segments = { idx: pointer_idx for idx, pointer_idx in enumerate(drainage_array) if pointer_idx not in [-1, -2, -3] and drainage_array[pointer_idx] == -2 }
    midpoints = { idx: list(regions_to_ridge_points[tuple(sorted([idx, pointer_idx]))]) for idx, pointer_idx in end_segments.items() }
    segments = {}

    for idx, pointer_idx in enumerate(drainage_array):
        if pointer_idx in [-1, -2, -3]:
            continue

        if idx in midpoints:
            end = midpoints[idx]
        else:
            end = pointer_idx
        segments[idx] = end

    return segments

def merge_adjacent_sinks(sinks, adjacency_graph):
    merged_sinks = {}
    visited = set()

    for sink in sinks:
        if sink in visited:
            continue

        component = set()
        queue = [sink]

        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            queue.extend(n for n in adjacency_graph[current] if n in sinks and n not in component)

        merged_sinks[sink] = component
    return merged_sinks

def label_basins(drainage_array, adjacency_graph, elevations, watersheds, mode=None):
    inland_sinks = sorted([idx for idx, val in enumerate(drainage_array) if val == -1], key=lambda i: elevations[i], reverse=False)
    watersheds_sink_labels = { sink : drainage_array[sink] for sink in np.unique(watersheds) if sink != -1 }

    DRAIN_INLAND = -1
    DRAIN_TO_OCEAN = -2

    membership_pointer = np.full(len(drainage_array), [-1], dtype=np.int32)
    labels = {
        sink:
        {
            "id": i,
            "parent": None,
            "spill_at": None,
            "spill_to": None,
            "escape": None,
            "members": []
        } for i, sink in enumerate(inland_sinks)
    }
    
    reconcile_basins = []

    for sink in inland_sinks:
        membership_pointer[sink] = sink

        current = sink
        visit_order = []
        current_water_elev = elevations[sink]

        while current:
            current_sink = watersheds[current]
            if watersheds_sink_labels[current_sink] == DRAIN_TO_OCEAN:
                _, idx = labels[sink]["members"][-1]
                labels[sink]["escape"] = idx
                labels[sink]["spill_to"] = current
            elif watersheds_sink_labels[current_sink] == DRAIN_INLAND and current_sink != sink:
                    if labels[current_sink]["escape"] is not None:
                        _, idx = labels[sink]["members"][-1]
                        labels[sink]["spill_at"] = idx
                        labels[sink]["spill_to"] = current
                        labels[sink]["escape"] = labels[current_sink]["escape"]
                        labels[sink]["parent"] = current_sink
                        break
            else:
                if current_water_elev < elevations[current] and labels[sink]["escape"] is None:
                    current_water_elev = elevations[current]
                elif labels[sink]["escape"] is not None:
                    current_water_elev = elevations[labels[sink]["escape"]]
                
                if elevations[current] <= current_water_elev:
                    membership_pointer[current] = sink
                    labels[sink]["members"].append((elevations[current], current))
                
                for neighbor in adjacency_graph[current]:
                    if (elevations[neighbor], neighbor) in labels[sink]["members"] or (elevations[neighbor], neighbor) in visit_order:
                        continue

                    heapq.heappush(visit_order, (elevations[neighbor], neighbor))

                _, idx = heapq.heappop(visit_order)
                current = idx

    # Some basins drain into basins above that haven't yet been explored. This step adds the escape cell once it has been identified.
    for sink in reconcile_basins:
        labels[sink]["escape"] = labels[labels[sink]["parent"]]["escape"]

    if mode == "debug":
        return labels, membership_pointer
    return labels, membership_pointer

def label_basins2(drainage_array, adjacency_graph, elevations, watersheds, mode=None):
    inland_sinks = sorted([idx for idx, val in enumerate(drainage_array) if val == -1], key=lambda i: elevations[i], reverse=False)
    watersheds_sink_labels = { sink : drainage_array[sink] for sink in np.unique(watersheds) if sink != -1 }

    DRAIN_INLAND = -1
    DRAIN_TO_OCEAN = -2

    labels = {
        sink:
        {
            "id": i,
            "parent": None,
            "spill_at_to": [],
            "escape": None,
            "members": []
        } for i, sink in enumerate(inland_sinks)
    }

    for sink in inland_sinks:

        current_water_level = elevations[sink]
        visit_order = []
        current = sink

        while current:
            current_sink = watersheds[current]
            if elevations[current] <= current_water_level:
                if watersheds_sink_labels == DRAIN_TO_OCEAN:
                    labels[sink]["escape"] = next(m for m in labels[sink]["members"] if current in adjacency_graph[m])
                    labels[sink]["spill_at_to"] = (labels[sink]["escape"], current)
                    break
                elif watersheds_sink_labels == DRAIN_INLAND and current_sink != sink:
                    if labels[current_sink]["escape"] is not None:
                        labels[sink]["escape"] = labels[current_sink]["escape"]
                        labels[sink]["spill_at_to"] = (next(m for m in labels[sink]["members"] if current in adjacency_graph[m]), current)
                        break

                labels[sink]["members"].append(current)

            for neighbor in adjacency_graph[current]:
                if neighbor in labels[sink]["members"] or (elevations[neighbor], neighbor) in visit_order:
                    continue

                heapq.heappush(visit_order, (elevations[neighbor], neighbor))

            _, idx = heapq.heappop(visit_order)
            current = idx
    
    return labels

def infer_hydrology(elevations_m, flow_graph_mm, basin_labels: dict, drainage_array, mode=None):
    sinks = sorted([sink for sink in basin_labels.keys()], key=lambda i: elevations_m[i], reverse=True)
    river_throughput = flow_graph_mm.copy()
    hydrology = {sink: {} for sink in sinks}
    if mode == 'debug':
        log = {sink: {} for sink in sinks}

    for sink in sinks:
        current_water_level = river_throughput[sink] / 1000 + elevations_m[sink]
        limit_idx = basin_labels[sink]["spill_at"] if basin_labels[sink]["spill_at"] is not None else basin_labels[sink]["escape"]
        members = []
        if len(basin_labels[sink]["members"]) > 1:
            members.append(sink)

        i = 1
        while current_water_level < elevations_m[limit_idx] and i <= len(basin_labels[sink]["members"]) - 1:
            _, idx = basin_labels[sink]["members"][i]

            if elevations_m[idx] <= current_water_level:
                current_water_level += river_throughput[idx] / 1000
                members.append(idx)

            i += 1

        if len(members) > 0:
            hydrology[sink]['members'] = members
            hydrology[sink]['lake_depth'] = [current_water_level - elevations_m[member] for member in members]
        
        if current_water_level > elevations_m[limit_idx]:
            hydrology[sink]['spillovers'] = (limit_idx, current_water_level - elevations_m[limit_idx])

        if hydrology[sink].get('spillovers'):
            delta = np.zeros(len(flow_graph_mm))
            idx, volume = hydrology[sink]['spillovers']
            current = idx
            while drainage_array[drainage_array[current]] not in [-2, -3] and drainage_array[current] != -1:
                delta[current] += volume
                current = drainage_array[current]

            river_throughput += delta

    return river_throughput, hydrology

def infer_hydrology2(elevations_m, flow_graph_mm, adjacency_graph, drainage_array, mode=None):
    inland_sinks = sorted([idx for idx, val in enumerate(drainage_array) if val == -1], key=lambda i: elevations_m[i], reverse=False)
    river_throughput = flow_graph_mm.copy()

    lakes = {}

    for sink in inland_sinks:
        current_water_level = river_throughput[sink] / 1000
        current = sink
        visit_order = []
        members = []
        while current:
            for neighbor in adjacency_graph:
                if neighbor in members or (elevations_m[neighbor], neighbor) in visit_order:
                    continue

                heapq.heappush(visit_order, (elevations_m[neighbor], neighbor))

            elev, next = heapq.heappop(visit_order)
            if elev < current_water_level + elevations_m[current]:
                current = next
                current_water_level = (current_water_level + elevations_m[current]) - (elev + river_throughput[next])
            else:
                current = None


        


def _compute_evaporation(normalized_temperatures, moisture, evaporation_factor=.175, temperature_strength=2):
    ev = (1 - moisture) * evaporation_factor * np.power(normalized_temperatures, temperature_strength)
    return ev