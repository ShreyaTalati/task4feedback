from ..task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from ..data import *
from ..device import *
from ..queue import *
from ..events import *
from ..resources import *
from ..task import *
from ..topology import *

from .state import SystemState

from ...types import Architecture, Device, TaskID, TaskState, TaskType, Time
from ...types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional, Callable, Type, Sequence
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict
from copy import copy, deepcopy

from rich import print


@dataclass(slots=True)
class SchedulerArchitecture:
    topology: InitVar[SimulatedTopology]
    completed_tasks: List[TaskID] = field(default_factory=list)

    def __post_init__(self, topology: SimulatedTopology):
        assert topology is not None

    def __getitem__(self, event: Event) -> Callable[[SystemState], Sequence[EventPair]]:
        try:
            function = getattr(self, event.func)
        except AttributeError:
            raise NotImplementedError(
                f"SchedulerArchitecture does not implement function {event.func} for event {event}."
            )

        def wrapper(scheduler_state: SystemState) -> Sequence[EventPair]:
            return function(scheduler_state, event)

        return wrapper

    def initialize(
        self, tasks: List[TaskID], scheduler_state: SystemState
    ) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def add_initial_tasks(self, task: SimulatedTask):
        pass

    def mapper(self, scheduler_state: SystemState, event: Event) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def reserver(
        self, scheduler_state: SystemState, event: Event
    ) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def launcher(
        self, scheduler_state: SystemState, event: Event
    ) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def complete_task(
        self, scheduler_state: SystemState, event: Event
    ) -> Sequence[EventPair]:
        return []

    def __str__(self):
        return f"SchedulerArchitecture()"

    def __repr__(self):
        self.__str__()


class SchedulerOptions:
    architecture_map: Dict[str, Type[SchedulerArchitecture]] = dict()
    state_map: Dict[str, Type[SystemState]] = dict()

    @staticmethod
    def register_architecture(scheduler_type: str) -> Callable[[Type], Type]:
        def decorator(cls):
            if scheduler_type in SchedulerOptions.architecture_map:
                raise ValueError(
                    f"Scheduler type {scheduler_type} is already registered."
                )
            SchedulerOptions.architecture_map[scheduler_type] = cls
            return cls

        return decorator

    @staticmethod
    def get_architecture(scheduler_type: str) -> Type[SchedulerArchitecture]:
        if scheduler_type not in SchedulerOptions.architecture_map:
            raise ValueError(
                f"Scheduler type `{scheduler_type}` is not registered. Registered types are: {list(SchedulerOptions.architecture_map.keys())}"
            )
        return SchedulerOptions.architecture_map[scheduler_type]

    @staticmethod
    def register_state(scheduler_type: str) -> Callable[[Type], Type]:
        def decorator(cls):
            if scheduler_type in SchedulerOptions.state_map:
                raise ValueError(
                    f"Scheduler State type {scheduler_type} is already registered."
                )
            SchedulerOptions.state_map[scheduler_type] = cls
            return cls

        return decorator

    @staticmethod
    def get_state(scheduler_type: str) -> Type[SystemState]:
        if scheduler_type not in SchedulerOptions.state_map:
            raise ValueError(
                f"Scheduler State type `{scheduler_type}` is not registered. Registered types are: {list(SchedulerOptions.state_map.keys())}"
            )
        return SchedulerOptions.state_map[scheduler_type]