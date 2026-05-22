# Geography Generator
A procedural geography generator built on a spherical Voronoi substrate, producing physically grounded terrain with elevation, climate, hydrology, and biome systems.
Overview
Rather than painting geography top-down, the generator propagates physical processes — tectonics, atmospheric circulation, drainage — across a graph of irregular Voronoi cells on a sphere, letting structure emerge from rules.
Pipeline
The generator is organized as a sequence of composable stages. Each stage consumes the cell graph and annotates it with new properties; later stages can read everything prior stages have written.
Substrate → Tectonics → Elevation → Wind & Climate → Hydrology → Biomes → Output
Substrate — A spherical Voronoi tessellation (via scipy.spatial.SphericalVoronoi) partitions the globe into irregular cells. Adjacency is derived directly from the Voronoi ridge structure, giving a graph where every edge carries a real geometric relationship. Using a sphere eliminates map-edge artifacts and preserves irregular cell geometry globally.
Tectonics — Plate boundaries are assigned and perturbed. Continental and oceanic crust designations drive the baseline elevation field, with boundary perturbation introducing mountain ranges and rift zones.
Elevation — Elevation is propagated outward from tectonic seeds, smoothed across the graph with configurable parameters controlling relief and continental shelf shape.
Wind & Climate — Prevailing wind vectors are computed per cell using latitude-based circulation bands. Temperature is derived from elevation and latitude; precipitation from orographic effects and wind exposure.
Hydrology — Drainage is resolved globally using a Priority-Flood algorithm over the elevation field. Flow accumulates downstream; basins are identified, and spill points between adjacent basins are located to reconstruct full river networks, including resolution of endorheic (internally draining) basins. Ocean cells are excluded from the hydrology graph.
Biomes — Cells are classified using a Whittaker-style temperature/precipitation lookup, producing discrete biome assignments.
Output — A Plotly-based 3D visualization with toggleable overlays: elevation, biomes, river flow, and wind vectors.
Data Model
Each cell carries:

Identity: index, centroid (unit sphere coordinates)
Topology: adjacency list, ridge lengths
Terrain: elevation, slope to each neighbor, plate assignment
Climate: temperature, precipitation, wind vector
Hydrology: drainage target cell, accumulated flow (m³/s)
Classification: biome, ocean/land flag

Design Principles

Graph-first: all computation operates on the adjacency graph, not a grid. This makes the spherical substrate a near drop-in replacement for any flat prototype.
Separation of topology and values: adjacency is computed once; all physical quantities are layered on top independently.
Sentinel-free schemas: absent data is represented by absence, not magic numbers.
Principled algorithms over special cases: where a process has global ordering requirements (drainage resolution), a globally ordered algorithm is used rather than locally patched heuristics.
