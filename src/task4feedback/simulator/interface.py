from .simulator import *
from ..types import *
from .data import *
from .device import *
from .preprocess import *

from typing import List, Dict, Set, Tuple, Optional, Callable, Type, Sequence
from .randomizer import Randomizer

from .rl.models.model import *
from .rl.models.env import *


@dataclass(slots=True)
class SimulatorConfig:
    topology: SimulatedTopology
    tasks: TaskMap
    data: DataMap
    scheduler_type: str = "parla"
    name: str = "Simulator"
    recorders: List[Type[Recorder]] = field(default_factory=list)
    simulated_tasks: SimulatedTaskMap = field(init=False)
    simulated_data: SimulatedDataMap = field(init=False)
    use_data: bool = True
    randomizer: Randomizer = field(default_factory=Randomizer)

    ###########################
    # RL related fields
    ###########################
    rl_env: RLBaseEnvironment = None
    rl_mapper: RLModel = None


def create_simulator(config: SimulatorConfig):
    simulated_data = create_data_objects(config.data, topology=config.topology)
    recorders = RecorderList(recorder_types=config.recorders)

    scheduler = SimulatedScheduler(
        topology=config.topology,
        scheduler_type=config.scheduler_type,
        recorders=recorders,
        randomizer=config.randomizer,
        rl_env=config.rl_env,
        rl_mapper=config.rl_mapper
    )

    tasklist, simulated_tasks = create_sim_graph(
        config.tasks, config.data, use_data=config.use_data
    )

    config.simulated_tasks = simulated_tasks
    config.simulated_data = simulated_data

    scheduler.register_datamap(simulated_data)
    scheduler.register_taskmap(simulated_tasks)

    scheduler.add_initial_tasks(tasklist, apply_sort=True)

    return scheduler
