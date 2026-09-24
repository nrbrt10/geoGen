from . import np
import plotly.graph_objects as go
from scipy.spatial import SphericalVoronoi
from types import SimpleNamespace as sn

terrain_colorscale =  [
        [0.0, 'darkblue'],
        [0.25, 'steelblue'],
        [0.5 - 1e-06, 'lightblue'],
        [0.5 + 1e-06, 'green'],
        [0.7, 'darkgreen'],
        [0.85, 'sienna'],
        [0.95, 'peru'],
        [1, 'white']
    ]

def polygons_to_triangles(polygons):
    i_idx, j_idx, k_idx = [], [], []

    for polygon in polygons:
        anchor = polygon[0]
        for n in range(1, len(polygon) -1):
            i_idx.append(anchor)
            j_idx.append(polygon[n])
            k_idx.append(polygon[n+1])

    return np.array(i_idx), np.array(j_idx), np.array(k_idx)

def map_regions_to_vertex(regions):
    vertices_map = {v: set() for region in regions for v in region}
    for idx, region in enumerate(regions):
        for vertex in region:
            vertices_map[vertex].add(idx)

    return vertices_map    

def map_intensity(regions, values):
    intensity = []
    for idx, region in enumerate(regions):
        num_triangles = len(region)
        intensity.extend([values[idx]] * num_triangles)
    
    return np.array(intensity)

def map_facecolors(regions, labels, color_map):
    colors = []
    for region_idx, region in enumerate(regions):
        num_triangles = len(region)
        label = labels[region_idx]
        if label == -1:
            continue
        r, g, b, a = color_map.get(label, [128, 128, 128, 1])
        colors.extend([f'rgb({r},{g},{b})'] * num_triangles)
    
    return colors

def normalize_elevations(elevations, sea_level, land_top_percentile=90):
    land_mask = elevations > sea_level
    ocean_mask = ~land_mask

    max_e = np.percentile(elevations, land_top_percentile)
    min_e = np.percentile(elevations, 1)

    normalized = np.zeros(len(elevations))
    normalized[ocean_mask] = np.clip((elevations[ocean_mask] - min_e) / (sea_level - min_e) * .5, 0, 0.5)
    normalized[land_mask] = np.clip(0.5 + (elevations[land_mask] - sea_level) / (max_e - sea_level) * 0.5, 0.5, 1)

    return normalized

def apply_terrain_to_vertices(regions, vertices, points, norm_elevations, scaling_factor=.1):
    topography_vertices = vertices.copy()
    topography_points = points.copy()
    vertices_by_region = map_regions_to_vertex(regions)
    displacement_coefficients = np.ones(len(np.concatenate((vertices, points))))
    vertices_offset = len(vertices)

    for vertex, regions in vertices_by_region.items():
        r = list(regions)
        magnitude = sum(norm_elevations[r] - 0.5) / len(r) * scaling_factor
        topography_vertices[vertex] = (vertices[vertex] * (1 + magnitude))
        displacement_coefficients[vertex] = 1 + magnitude

    for idx, point in enumerate(points):
        magnitude = (norm_elevations[idx] - 0.5) * scaling_factor
        topography_points[idx] = point * (1 + magnitude)
        displacement_coefficients[idx+vertices_offset] = 1 + magnitude

    return np.concatenate((topography_vertices, topography_points)), vertices_offset, displacement_coefficients

def get_triangles_w_centroids(regions, offset):
    i_idx, j_idx, k_idx = [], [], []
    triangle_to_regions_map = {}

    i = 0
    for idx, region in enumerate(regions):
        triangle_to_regions_map[idx] = []
        anchor = idx+offset
        for n in range(0, len(region)):
            i_idx.append(anchor)
            j_idx.append(region[n])
            k_idx.append(region[(n+1) % len(region)])
            triangle_to_regions_map[idx].append(i)
            i += 1

    return (np.array(i_idx), np.array(j_idx), np.array(k_idx)), triangle_to_regions_map

def create_sphere_topography(sv: SphericalVoronoi, normalized_elevations):
    terrain_vertices, offset, displacement = apply_terrain_to_vertices(sv.regions, sv.vertices, sv.points, normalized_elevations)
    terrain_triangles, triangle_to_region_map = get_triangles_w_centroids(sv.regions, offset)

    return sn(
        vertices=terrain_vertices,
        triangles=terrain_triangles,
        vertices_offset=offset,
        vertices_displacement=displacement,
        triangle_to_region_map=triangle_to_region_map
        )

def create_terrain_mesh(vertices, triangles, customdata_template, custom_data, scale=1):
    x, y, z = vertices.T * scale
    i, j, k = triangles

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        flatshading=True,
        customdata=custom_data,
        hovertemplate=customdata_template
    )

def create_water_level_mesh(vertices, regions, scale=1):
    x, y, z = vertices.T * scale
    i, j, k = polygons_to_triangles(regions)
    return  go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='steelblue',
            opacity=0.4,
            flatshading=True,
            showscale=False
        )

def create_subset_mesh(
        regions,
        vertices,
        triangles,
        triangles_to_region,
        *,
        values,
        colorscale=None,
        facecolor=None,
        scale=1,
        z_displacement=1.0001,
        opacity=.25,
        customdata_template=None,
        customdata=None,
        invalid_tag=-1
        ):
    x, y, z = vertices.T * z_displacement * scale
    i, j, k = triangles
    mask = np.array([idx for indices in np.where(values != invalid_tag)[0] for idx in triangles_to_region[indices]])
    
    if colorscale is not None:
        intensity = map_intensity(regions, values)[mask]

        return go.Mesh3d(
            x=x, y=y, z=z,
            i=i[mask], j=j[mask], k=k[mask],
            colorscale=colorscale,
            intensity=intensity,
            intensitymode='cell',
            opacity=opacity,
            flatshading=True,
            showscale=False,
            customdata=customdata[mask],
            hovertemplate=customdata_template
        )
    
    if facecolor is not None:
        colors = map_facecolors(regions, values, facecolor)

        return go.Mesh3d(
            x=x, y=y, z=z,
            i=i[mask], j=j[mask], k=k[mask],
            facecolor=colors,
            opacity=opacity,
            flatshading=True,
            showscale=False,
            customdata=customdata[mask],
            hovertemplate=customdata_template
        )

def apply_colorscale(mesh, regions, values, colorscale):
    
    trace = go.Mesh3d(mesh)
    trace.intensity = map_intensity(regions, values)
    trace.colorscale = colorscale
    trace.intensitymode = 'cell'
    
    return trace

def render_world(sv: SphericalVoronoi, sphere_topography, customdata_template, customdata, mesh_layers: list, overlay_traces, scale=1, display_water_level=True):
    fig = go.Figure()

    base_mesh = create_terrain_mesh(sphere_topography.vertices, sphere_topography.triangles, customdata_template, customdata, scale)
    fig.add_trace(base_mesh)

    for name, trace in overlay_traces:
        try:
            if isinstance(trace, list):
                fig.add_traces(trace)
            else:
                fig.add_trace(trace)
        except Exception as e:
            print(f'Exception encountered while creating mesh {name}: \n', e)

    mesh_buttons = [
        dict(label=name, method='restyle', args=[{'intensity': [map_intensity(sv.regions, intensity)], 'colorscale': [colorscale], 'intensitymode': ['cell']}, [0]]) for name, intensity, colorscale in mesh_layers
    ]

    overlay_buttons = [
        dict(label=name, method='restyle', args=[{'visible': True}, [i+1]], args2=[{'visible': False}, [i+1]]) for i, (name, _) in enumerate(overlay_traces)
    ]

    if display_water_level:
        water_idx = len(overlay_traces) + 1
        fig.add_trace(create_water_level_mesh(sv.vertices, sv.regions, scale))
        overlay_buttons += [dict(label='Water', method='restyle', args=[{'visible': True}, [water_idx]], args2=[{'visible': False}, [water_idx]])]

    fig.update_layout(
        updatemenus=[
            dict(type='dropdown', y=1.1, buttons=mesh_buttons),
            dict(type='buttons', y=1.0, buttons=overlay_buttons)
    ])

    fig.show(renderer='browser')
    fig.write_html('world.html')