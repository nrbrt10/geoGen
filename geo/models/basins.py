from . import dataclass, field
from . import List, Optional
from enum import StrEnum, auto

@dataclass
class SpillData():
    at: int
    to: int

@dataclass
class Basin():
    id: int
    capacity: Optional[float] = None
    spill: SpillData | None = None
    saddle: float | None = None
    children: List[tuple[float, 'Basin']] = field(default_factory=list)
    members: List[int] = field(default_factory=list)

    def set_spill(self, at, to):
        self.spill = SpillData(at=at, to=to)

class PoolState(StrEnum):
    SPILLED = auto()
    SUBMERGED = auto()
    UNPROCESSED = auto()
    MERGED = auto()

@dataclass
class BasinPool():
    id: int
    parent: Optional[int] = None
    state: PoolState = PoolState.UNPROCESSED
    elevation: float = None
    outflow: float = None
    members: List[int] = field(default_factory=list)

    def merge_to(self, parent):
        self.parent = parent
        self.state = PoolState.MERGED

    def set_elevation(self, elevation):
        self.elevation = elevation

    def set_outflow(self, outflow):
        self.outflow = outflow
        self.state = PoolState.SPILLED

    def add_child(self, child):
        self.children.append(child)