from ..types import TaskID, TaskInfo, TaskState, Optional
from dataclasses import dataclass, field


@dataclass()
class Event:
    func: str
    time: Optional[float] = None

    def __eq__(self, other):
        return self.func == other.func and self.time == other.time

    def __lt__(self, other):
        return self.time < other.time

    def __hash__(self):
        return hash((self.func, self.time))


@dataclass()
class PhaseEvent(Event):
    max_tasks: int | None = None


@dataclass()
class TaskEvent(Event):
    task: TaskID = field(default_factory=TaskID)


@dataclass()
class Mapper(PhaseEvent):
    func: str = "mapper"


@dataclass()
class Reserver(PhaseEvent):
    func: str = "reserver"


@dataclass()
class Launcher(PhaseEvent):
    func: str = "launcher"


@dataclass()
class TaskCompleted(TaskEvent):
    func: str = "complete_task"
