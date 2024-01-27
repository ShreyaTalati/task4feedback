from ..types import Architecture, Device, TaskID, TaskState, ResourceType
from dataclasses import dataclass, field
from .queue import *
from .datapool import *
from enum import IntEnum
from typing import List, Dict, Set, Tuple, Optional, Self, Type
from fractions import Fraction
from decimal import Decimal
from collections import defaultdict as DefaultDict

from .eviction.base import EvictionPool
from .eviction.lru import LRUEvictionPool


Numeric = int | float | Fraction | Decimal


@dataclass(slots=True, init=False)
class ResourceSet:
    store: DefaultDict[ResourceType, Numeric] = field(
        default_factory=lambda: DefaultDict(int)
    )

    def __init__(self, vcus: Numeric, memory: int, copy: int):
        self.store = DefaultDict(int)

        self.store[ResourceType.VCU] = Fraction(vcus)
        self.store[ResourceType.MEMORY] = memory
        self.store[ResourceType.COPY] = copy

    def __getitem__(self, key: ResourceType) -> Numeric:
        return self.store[key]

    def __setitem__(self, key: ResourceType, value: Numeric):
        self.store[key] = value

    def __iter__(self):  # For unpack operator
        return iter(self.store)

    def add_types(self, other: Self, resource_types: List[ResourceType]) -> Self:
        for key in resource_types:
            if key in other.store and key in self.store:
                self.store[key] += other.store[key]
        return self

    def add_all(self, other: Self) -> Self:
        for key in self.store:
            if key in other.store and key in self.store:
                self.store[key] += other.store[key]
        return self

    def subtract_types(self, other: Self, resource_types: List[ResourceType]) -> Self:
        for key in resource_types:
            if key in other.store and key in self.store:
                self.store[key] -= other.store[key]
        return self

    def subtract_all(self, other: Self) -> Self:
        for key in self.store:
            if key in other.store and key in self.store:
                self.store[key] -= other.store[key]
        return self

    def verify(self, max_resources: Optional[Self] = None):
        for key in self.store:
            if self.store[key] < 0:
                raise ValueError(
                    f"ResourceSet {self} contains negative value for {key}"
                )

        if max_resources is not None:
            for key in self.store:
                if self.store[key] > max_resources.store[key]:
                    raise ValueError(
                        f"ResourceSet {self} exceeds maximum resources {max_resources}"
                    )

    def __str__(self) -> str:
        return f"ResourceSet({self.store})"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: Self) -> bool:
        for key in other.store:
            if self.store[key] > other.store[key]:
                return False
        return True

    def __le__(self, other: Self) -> bool:
        for key in other.store:
            if self.store[key] >= other.store[key]:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Self):
            for key in other.store:
                if self.store[key] != other.store[key]:
                    return False
            return True
        else:
            return False


@dataclass(slots=True)
class SimulatedDevice:
    name: Device
    resources: ResourceSet
    datapool: DataPool = field(default_factory=DataPool)
    eviction_pool_type: Type[EvictionPool] = LRUEvictionPool
    eviction_pool: EvictionPool = field(init=False)

    def __post_init__(self):
        self.eviction_pool = self.eviction_pool_type()

    def __str__(self) -> str:
        return f"Device({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __getitem__(self, key: ResourceType) -> Numeric:
        return self.resources[key]

    def add_data(self, data: SimulatedData):
        self.datapool.add(data)

    def remove_data(self, data: SimulatedData):
        self.datapool.remove(data)

    def add_evictable(self, data: SimulatedData):
        self.eviction_pool.add(data)

    def remove_evictable(self, data: SimulatedData):
        self.eviction_pool.remove(data)
