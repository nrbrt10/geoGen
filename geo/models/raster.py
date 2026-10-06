from . import dataclass, field
from . import List, Optional

@dataclass
class HydrologyData():
    lake_id: float
    depth_m: float



@dataclass
class RasterDTO():
    id: int
    continent_id: int
    centroid: list[float]
    vertices: list[list[float]]
    adjacency: list[int]
    elevation_m: float
    temperature_c: float
    precipitation_mmy: float
    throughput_m3s: float | None
    hydrology: Optional[HydrologyData] = None

    def is_lake(self, lake_id, depth_m):
        self.hydrology = HydrologyData(lake_id=lake_id, depth_m=depth_m)
