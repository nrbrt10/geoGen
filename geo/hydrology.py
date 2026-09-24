from . import np, deque, heapq, math, sn
from .helpers import clamp
from .adjacency import sort_adjacency_graph
from .graph_utils import find_root
from .models.basins import Basin, BasinPool, PoolState
from enum import IntEnum

class HydrologyTag(IntEnum):
    INLAND_SINK = -1
    OCEAN_OUTLET = -2
    OCEAN = -3

def generate_drainage_data(elevations: np.array, adjacency: dict, sea_level: float, base_absorption: float=.4):

    drainage = np.full(len(elevations), HydrologyTag.OCEAN, dtype=np.int32)
    absorption = np.full(len(elevations), [-1], dtype=np.float64)
    slopes = np.full(len(elevations), [-1], dtype=np.float32)

    sorted_adjacency = sort_adjacency_graph(adjacency, elevations)

    for idx, elevation in enumerate(elevations):
        if elevation <= sea_level:
            continue

        lowest = next((neighbor for neighbor in sorted_adjacency[idx] if elevations[neighbor] < elevation), None)
        slope = 1 if lowest is None else elevation - elevations[lowest]
            
        if lowest is None:
            drainage[idx] = HydrologyTag.INLAND_SINK
            slopes[idx] = 1
        elif elevations[lowest] <= sea_level:
            drainage[idx] = HydrologyTag.OCEAN_OUTLET
            slopes[idx] = slope
        else:
            drainage[idx] = lowest
            slopes[idx] = slope
            
        absorption[idx] = base_absorption * clamp(1/slope, 0, 1) ** 1.75
        
    return drainage, absorption, slopes

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
    watersheds_array = np.full(len(drainage_array), -1, dtype=np.int32)

    for sink in sinks:
        stack = [sink]
        while stack:
            current = stack.pop()

            watersheds_array[current] = sink

            if current in inverted_drainage_array:
                stack.extend(inverted_drainage_array[current])
    
    return watersheds_array

def drainage_dependencies(inverted_drainage_graph: dict) -> dict:
    dependencies_graph = {}
    for idx, dependencies in inverted_drainage_graph.items():
        dependencies_graph[idx] = len(dependencies)

    return dependencies_graph

def compute_flow_volume(
        drainage_graph: np.array,
        rainfall_mmy: np.array,
        absorption: np.array,
        slopes: np.array,
        soil_capacity: np.array,
        areas_km2: np.array,
        evaporation_rate: np.array | float=.175,
        ):

    sources = [idx for idx, val in enumerate(drainage_graph) if idx not in drainage_graph and val not in [HydrologyTag.OCEAN]]
    dependency_count = drainage_dependencies(invert_drainage_array(drainage_graph))
    inflows = np.zeros(shape=len(drainage_graph))
    saturation = np.full(len(drainage_graph), -1, dtype=np.float64)
    areas_m2 = areas_km2 * 1e6
    rainfall_m3 = rainfall_mmy / 1000 * areas_m2
    drainage_volumes = np.full(len(drainage_graph), np.nan)

    if isinstance(evaporation_rate, float):
        evaporation_rate = np.full(len(drainage_graph), evaporation_rate)

    queue = deque()
    queue.extend(sources)

    while queue:
        current = queue.popleft()

        if drainage_graph[current] in [-3]:
            continue

        if slopes[current] == -1 or absorption[current] == -1:
            raise ValueError(f'Slopes or absorption data invalid at: {current}')

        total_load = rainfall_m3[current] + inflows[current]
        saturation[current] = np.clip(math.array_safe_divide(total_load * absorption[current] * (1 - evaporation_rate[current]), soil_capacity[current], 0), 0, 2)
        runoff = total_load * (1 - evaporation_rate[current]) * (1 - absorption[current])
        drainage_volumes[current] = runoff

        if drainage_graph[current] in [-1, -2]:
            continue

        inflows[drainage_graph[current]] += runoff
        dependency_count[drainage_graph[current]] -= 1

        if dependency_count[drainage_graph[current]] == 0:
            queue.append(drainage_graph[current])

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

def label_basins_se(elevations, drainage_array, watersheds, adjacency_graph) -> tuple[np.array, dict[int, Basin]]:
    '''
    
    '''

    sorted_land_adjacency = sort_adjacency_graph(adjacency_graph, elevations)
    inland_sinks = np.array([i for i, v in enumerate(drainage_array) if v == HydrologyTag.INLAND_SINK])
    drain_to_sea = drainage_array[watersheds] == HydrologyTag.OCEAN_OUTLET

    edges = set()
    for u in sorted_land_adjacency:
        for v in sorted_land_adjacency[u]:
            if drain_to_sea[u] and drain_to_sea[v]:
                continue
            if u < v:
                edges.add((max(elevations[u], elevations[v]), min(elevations[u], elevations[v]), u, v))

    sorted_edges = sorted(list(edges), key=lambda x: (x[0], x[1]))

    spill_data = {}
    spilled = set()
    basin_data = {sink: Basin(id=sink) for sink in inland_sinks}

    basin_members = np.full(len(drainage_array), -1, dtype=np.int32)
    basin_members[inland_sinks] = inland_sinks

    for i, (saddle, _, u, v) in enumerate(sorted_edges):

        ws_u = watersheds[u]
        ws_v = watersheds[v]

        root_u = ws_u if ws_u not in spilled else find_root(basin_members, ws_u)
        root_v = ws_v if ws_v not in spilled else find_root(basin_members, ws_v)
            
        if root_u in spilled and root_v in spilled:
            continue

        if drain_to_sea[u] or drain_to_sea[v]:
            if drain_to_sea[u] and root_v not in spilled:
                sink = root_v
                member = v
                spill_at = v
                spill_to = u
            elif drain_to_sea[v] and root_u not in spilled:
                sink = root_u
                member = u
                spill_at = u
                spill_to = v
            else:
                continue

            spilled.add(sink)
            basin_data[sink].set_spill(at=spill_at, to=spill_to)
            basin_data[sink].saddle = saddle
            basin_members[member] = sink
            continue

        if root_u == root_v:
            if root_u not in spill_data:
                basin_members[v] = root_v
                basin_members[u] = root_u
            continue

        if root_u in spilled and root_v not in spilled:
            parent = root_u
            basin_members[v] = root_v
            sink = root_v
            spill_at = v
            spill_to = u

        elif root_v in spilled and root_u not in spilled:
            parent = root_v
            basin_members[u] = root_u
            sink = root_u
            spill_at = u
            spill_to = v

        else:
            basin_members[v] = root_v
            basin_members[u] = root_u
            if elevations[root_v] >= elevations[root_u]:
                parent = root_u
                sink = root_v
                spill_at = v
                spill_to = u
            else:
                parent = root_v
                sink = root_u
                spill_at = u
                spill_to = v
            basin_members[sink] = parent
            basin_data[parent].children.append((saddle, sink))

        basin_data[sink].set_spill(at=spill_at, to=spill_to)
        basin_data[sink].saddle = saddle
        spilled.add(sink)

    for i, root in enumerate(basin_members):
        if root != -1:
            basin_members[i] = find_root(basin_members, root)

    return basin_members, basin_data

def get_basin_drainage_links(basin_data, watersheds):
    links = {}
    for sink, data in basin_data.items():
        spill_to = data.spill.to
        to_root = watersheds[spill_to]
        if to_root not in basin_data:
            continue
        links[sink] = to_root

    return links

def get_basin_dependencies(basin_data, basin_links):
    hierarchy = {sink: set() for sink in basin_data.keys()}
    for at_root, to_root in basin_links.items():
        hierarchy[to_root].add(at_root)

    return hierarchy

def cascade_throughputs_djs(start: int, value: float, pointer_array: np.array, djs_parents: np.array) -> np.array:
    '''Returns delta array to sum with throughputs'''
    delta = np.zeros(len(pointer_array))
    cur = start
    while True:
        if djs_parents[cur] != -1:
            root = find_root(djs_parents, cur)
            delta[root] += value
            break

        delta[cur] += value
        if pointer_array[cur] in [HydrologyTag.OCEAN_OUTLET, HydrologyTag.INLAND_SINK]:
            break
        
        cur = pointer_array[cur]

    return delta

def basin_pooling_solver(
        basin_id: int,
        basin_data: dict[int, Basin],
        djs_parents: np.array[int],
        elevations_m: np.array[float],
        depths_m: np.array[float],
        areas_m2: np.array[float],
        throughputs_m3: np.array[float],
        pool_data: dict[int, BasinPool],
        hydrology_graph: np.array[int],
        watersheds: np.array[int]
        ):

    basin = basin_data[basin_id]
    members = basin.members
    saddle_elev = basin.saddle

    volume_in_basin = sum(depths_m[members] * areas_m2[members])
    total_actual_capacity = basin.capacity - volume_in_basin
    inflow_vol = throughputs_m3[basin_id]

    # Total volumetric check
    if (outflow := inflow_vol - total_actual_capacity) > 0:
        actual_members = [i for i in members if elevations_m[i] < saddle_elev]

        hydrology_graph[basin.spill.at] = basin.spill.to
        throughputs_m3 += cascade_throughputs_djs(basin.spill.to, outflow, hydrology_graph, djs_parents)

        depths_m[actual_members] += saddle_elev - (elevations_m[actual_members] + depths_m[actual_members])
        djs_parents[actual_members] = basin_id

        pool_data[basin_id].set_outflow(outflow)
        pool_data[basin_id].elevation = saddle_elev
        pool_data[basin_id].members = actual_members

        return outflow

    # Pooling logic
    else:
        
        cum_areas = np.cumsum(areas_m2[members])
        accum_elevations = np.maximum.accumulate(elevations_m[members])
        cum_el_x_area = np.cumsum(areas_m2[members] * elevations_m[members])
        cum_capacity = accum_elevations * cum_areas - cum_el_x_area
        members_current = []

        for i, member in enumerate(members):
            volume_in_basin_at_i = sum(depths_m[members_current] * areas_m2[members_current])
            actual_capacity = cum_capacity[i] - volume_in_basin_at_i
            cur_elevation = accum_elevations[i]

            if inflow_vol > actual_capacity:
                outflow = inflow_vol - actual_capacity
                member_root = watersheds[member] if djs_parents[member] == -1 else find_root(djs_parents, member)
                
                # When the basin touches another watershed, it either escaped (impossible in this loop), or touched a nested basin.
                if member_root != basin_id:
                    if pool_data[member_root].state == PoolState.UNPROCESSED:
                        raise ValueError(f'Pool {member_root} elevation is None at: {basin_id}')
                    
                    elif pool_data[member_root].elevation == cur_elevation:
                        parent, child = (basin_id, member_root) if elevations_m[basin_id] < elevations_m[member_root] else (member_root, basin_id)
                        djs_parents[child] = parent
                        members_current.append(member)

                        if basin_id == child:
                            pool_data[child].merge_to(parent)
                            return outflow
                        else:
                            pool_data[child].merge_to(parent)
                        
                        continue
                    
                    throughputs_m3 += cascade_throughputs_djs(member, outflow, hydrology_graph, djs_parents)
                    outflow = basin_pooling_solver(member_root
                                                   , basin_data
                                                   , djs_parents
                                                   , elevations_m
                                                   , depths_m
                                                   , areas_m2
                                                   , throughputs_m3
                                                   , pool_data
                                                   , hydrology_graph
                                                   , watersheds)
                    
                else:
                    djs_parents[member] = basin_id
                    members_current.append(member)
            else:
                prev_elev = next((e for e in accum_elevations[::-1] if e < cur_elevation), None)
                if prev_elev is None:
                    raise ValueError(f'Could not find a value lower than {cur_elevation} in {accum_elevations}')

                prev_actual_capacity = cum_capacity[i-1] - volume_in_basin_at_i
                prev_cum_area = cum_areas[i-1]
                outflow_m = (inflow_vol - prev_actual_capacity) / prev_cum_area
                final_elevation = outflow_m + prev_elev
                depths_m[members_current] += final_elevation - (elevations_m[members_current] + depths_m[members_current])

                pool_data[basin_id].elevation = final_elevation
                pool_data[basin_id].state = PoolState.SUBMERGED
                
                outflow = 0
                break

        if outflow > 0:
            if max(elevations_m[members]) < saddle_elev:
                total_area = cum_areas[i]
                outflow_m = (inflow_vol - actual_capacity) / total_area
                final_elevation = outflow_m + cur_elevation
                depths_m[members_current] += final_elevation - (elevations_m[members_current] + depths_m[members_current])

                pool_data[basin_id].elevation = final_elevation
                pool_data[basin_id].state = PoolState.SUBMERGED
                
            else:
                raise Exception(basin_id, saddle_elev, elevations_m[members])

    return

def infer_hydrology(
        elevations_m: np.array,
        flow_graph_m3: np.array,
        polygon_areas_km2: np.array,
        basin_data: dict[int, Basin],
        drainage_graph: np.array,
        watersheds: np.array,
        ):
    
    areas_m2 = polygon_areas_km2 * 1e6

    hydrology_graph = drainage_graph.copy()
    throughputs_m3 = flow_graph_m3.copy()
    depths_m = np.zeros(len(drainage_graph), dtype=np.float64)

    basin_links = get_basin_drainage_links(basin_data, watersheds)
    basin_dependencies = get_basin_dependencies(basin_data, basin_links)
    dependecy_count = {id: len(depedencies) for id, depedencies in basin_dependencies.items() if len(depedencies) > 0}

    leaves = [(basin_data[id].saddle, id)
              for id, dependencies in basin_dependencies.items()
              if len(dependencies) == 0]

    # DJS init
    parents = np.full(len(drainage_graph), -1, dtype=np.int32)
    parents[list(basin_data.keys())] = list(basin_data.keys())
    pool_data = {basin: BasinPool(id=basin) for basin in basin_data}

    queue = deque()
    queue.extend(leaves)

    while queue:
        _, current = queue.popleft()

        basin_pooling_solver(current, basin_data, parents, elevations_m, depths_m, areas_m2, throughputs_m3, pool_data, hydrology_graph, watersheds)

        if current not in basin_links:
            continue
        dependant = basin_links[current]
        dependecy_count[dependant] -= 1

        if dependecy_count[dependant] == 0:
            data = basin_data[dependant]
            queue.append((data.saddle, dependant))

    return hydrology_graph, throughputs_m3, parents, depths_m, pool_data

def _compute_evaporation(normalized_temperatures, moisture, evaporation_factor=.175, temperature_strength=2):
    ev = (1 - moisture) * evaporation_factor * np.power(normalized_temperatures, temperature_strength)
    return ev