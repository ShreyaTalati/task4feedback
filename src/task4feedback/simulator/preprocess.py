from ..types import *
from ..load import *
from .task import *
from .data import *
from .device import *
import networkx as nx
from .topology import *
from copy import deepcopy
import random


def summarize_dependencies(taskmap: TaskMap | SimulatedTaskMap):
    @dataclass(slots=True)
    class DependencySummary:
        read: List[DataID]
        write: List[DataID]
        read_write: List[DataID]
        depends_on: List[TaskID]

    summarized = {}

    for task in taskmap.values():
        if isinstance(task, SimulatedTask):
            task = task.info

        read = [d.id for d in task.data_dependencies.read]
        write = [d.id for d in task.data_dependencies.write]
        read_write = [d.id for d in task.data_dependencies.read_write]
        depends_on = task.dependencies

        summarized[task.id] = DependencySummary(read, write, read_write, depends_on)

    return summarized


def data_from_task(task: TaskInfo, access: AccessType) -> List[DataID]:
    return [d.id for d in task.data_dependencies[access]]


def find_writer_bfs(
    graph: TaskMap, node: TaskID, target: DataID, verbose: bool = True
) -> List[TaskID | DataID]:
    """
    Return last task to touch the data.
    @param graph: TaskMap (TaskID -> TaskInfo)
    @param node: TaskID to start search from
    @param target: DataID to search for
    """

    queue = []
    visited = []
    visited.append(node)
    queue.append(node)
    found = []

    while queue:
        s = queue.pop(0)
        if verbose:
            print(f"Checking dependencies of {s}")
        for neighbor_id in graph[s].dependencies:
            neighbor = graph[neighbor_id]

            writes_to = data_from_task(neighbor, AccessType.WRITE)
            writes_to = writes_to + data_from_task(neighbor, AccessType.READ_WRITE)
            write_to = set(writes_to)

            if verbose:
                print(f"Dependency {neighbor_id} writes to {writes_to}")

            if target in writes_to:
                if verbose:
                    print(f"Found writer {neighbor_id} to {target}")
                found.append(neighbor_id if neighbor_id != node else target)

            if neighbor_id not in visited and len(found) == 0:
                visited.append(neighbor.id)
                queue.append(neighbor.id)

    if verbose and len(found) == 0:
        print(f"Could not find writer to {target} from {node}.")
    return found


DataWriter = Dict[DataID, List[TaskID | DataID]]
DataWriters = Dict[TaskID, DataWriter]


def most_recent_writer(
    graph: TaskMap, task: TaskInfo, verbose: bool = False
) -> DataWriter:
    """
    For each of a tasks inputs, return the most recent writer.
    If this is the first task to write the data, return the DataID itself.
    """

    read_data = data_from_task(task, AccessType.READ)
    read_write_data = data_from_task(task, AccessType.READ_WRITE)

    read_data = read_data + read_write_data
    touches = set(read_data)

    if verbose:
        print(f"Task {task.id} reads data: {touches}")

    recent_writer = dict()

    for target in touches:
        if verbose:
            print(f"Looking for most recent writer to Data {target}")
        recent_writer[target] = find_writer_bfs(graph, task.id, target, verbose=verbose)

    return recent_writer


def find_recent_writers(graph: TaskMap, verbose: bool = False) -> DataWriters:
    """
    For each task, find the most recent writer for each of its inputs.
    """
    recent_writers = dict()

    if verbose:
        print("Finding recent writers...")

    for task in graph.values():
        if verbose:
            print(f"Looking at data from task: {task.id}")
        recent_writers[task.id] = most_recent_writer(graph, task, verbose=verbose)

    return recent_writers


def create_compute_tasks(graph: TaskMap) -> SimulatedComputeTaskMap:
    """
    Create compute tasks for each task in the graph.
    """
    compute_tasks = dict()

    for task in graph.values():
        compute_task = SimulatedComputeTask(task.id, task)
        filter_data_dependenices(compute_task)
        compute_tasks[task.id] = compute_task

    return compute_tasks


def create_data_tasks(
    graph: SimulatedComputeTaskMap, recent_writers: DataWriters
) -> SimulatedDataTaskMap:
    """
    Create data tasks for each data item in the task.
    """
    data_tasks = dict()

    for task in graph.values():
        task_info = task.info
        recent_writer = recent_writers[task_info.id]
        for i, (data, writer_list) in enumerate(recent_writer.items()):
            # print(f"Creating data task for {data} from {writer}")
            dependencies = writer_list

            data_task_id = TaskID(taskspace=f"{task_info.id}.data", task_idx=data.idx)

            runtime = TaskPlacementInfo()
            runtime.add(Device(Architecture.ANY, -1), TaskRuntimeInfo())
            data_info = TaskDataInfo(read=[DataAccess(id=data, device=0)])

            data_task_info = TaskInfo(
                id=data_task_id,
                dependencies=dependencies,
                runtime=runtime,
                data_dependencies=data_info,
            )

            data_task = SimulatedDataTask(
                name=data_task_id, info=data_task_info, parent=task.name
            )
            data_task.local_index = 0
            data_tasks[data_task_id] = data_task
            task.add_data_dependency(data_task_id)

    return data_tasks


def filter_data_dependenices(task: SimulatedTask):
    data_info = task.info.data_dependencies
    read = data_info.read
    write = data_info.write
    read_write = data_info.read_write

    # Remove read-write dependencies from read and write
    # All sets must be disjoint

    read_set = set([d.id for d in read]).difference([d.id for d in read_write])
    write_set = set([d.id for d in write]).difference([d.id for d in read_write])

    assert (
        len(read_set.intersection(write_set)) == 0
    ), "Read and write sets must be disjoint"

    read = list(DataAccess(id=d) for d in read_set)
    write = list(DataAccess(id=d) for d in write_set)

    data_info.read = read
    data_info.write = write


def create_task_graph(graph: TaskMap) -> SimulatedComputeTaskMap:
    """
    Create a task graph from a task map.
    """
    compute_tasks = create_compute_tasks(graph)
    return compute_tasks


def create_data_task_graph(
    graph: TaskMap, compute_tasks: SimulatedComputeTaskMap, verbose: bool = False
) -> SimulatedDataTaskMap:
    recent_writers = find_recent_writers(graph, verbose=False)
    data_tasks = create_data_tasks(compute_tasks, recent_writers)
    return data_tasks


def combine_task_graphs(
    graph1: SimulatedTaskMap, graph2: SimulatedTaskMap
) -> SimulatedTaskMap:
    """
    Combine two task graphs into one.
    """
    graph = dict()
    graph.update(graph1)
    graph.update(graph2)
    return graph


def create_sim_graph(
    tasks: TaskMap, data: DataMap, use_data: bool = True
) -> Tuple[List[TaskID], SimulatedTaskMap]:
    compute_tasks = create_task_graph(tasks)
    if use_data:
        data_tasks = create_data_task_graph(tasks, compute_tasks)
        taskmap = combine_task_graphs(compute_tasks, data_tasks)
    else:
        taskmap: SimulatedTaskMap = compute_tasks

    tasklist = list(compute_tasks.keys())
    populate_dependents(taskmap)
    # compute_depths(taskmap)

    return tasklist, taskmap


def read_sim_graph(
    graph_name: str, use_data: bool = False
) -> Tuple[List[TaskID], SimulatedTaskMap, DataMap]:
    tasks = read_tasks_from_yaml(graph_name)
    data = read_data_from_yaml(graph_name)

    tasklist, taskmap = create_sim_graph(tasks, data, use_data)

    return tasklist, taskmap, data


def build_networkx_graph(
    tasks: SimulatedTaskMap,
) -> Tuple[nx.DiGraph, Dict[TaskID, str]]:
    G = nx.DiGraph()
    labels = {}

    color_map = ["red", "blue"]

    for name, info in tasks.items():
        color_idx = 0 if isinstance(info, SimulatedComputeTask) else 1
        color = color_map[color_idx]

        G.add_node(name, label=name, color=color, info=info)
        for dependency in info.dependencies:
            G.add_edge(dependency, name, color=color)

        labels[name] = str(name)

    return G, labels


def convert_devices_to_list(devices: Optional[Devices]) -> List[Device]:
    if devices is None:
        # Default to CPU 0
        devices = Device(Architecture.CPU, 0)

    if isinstance(devices, Device):
        return [devices]
    else:
        return list(devices)


def create_data_objects(
    datamap: DataMap, topology: SimulatedTopology
) -> SimulatedDataMap:
    data_objects = dict()

    devices = topology.devices
    devices = [device.name for device in devices]
    print(len(datamap))
    i = 0
    for data_info in datamap.values():
        print(f'In data_info loop: {i} :: {data_info.id} ')
        data_objects[data_info.id] = SimulatedData(
            system_devices=devices, info=data_info
        )
        i += 1
    return data_objects


def apply_networkx_order(G: nx.DiGraph, tasks: SimulatedTaskMap) -> List[TaskID]:
    """
    Sort a graph by a topology, and return a valid order of the graph.
    """
    import networkx as nx

    nodes = list(nx.topological_sort(G))

    for i, node in enumerate(nodes):
        tasks[node].info.order = i

    return nodes


def topological_sort(tasklist: List[TaskID], taskmap: SimulatedTaskMap) -> List[TaskID]:
    G, labels = build_networkx_graph(taskmap)
    apply_networkx_order(G, taskmap)

    return sort_tasks_by_order(tasklist, taskmap)


def sort_tasks_by_order(tasklist: List[TaskID], taskmap: SimulatedTaskMap):
    """
    Sort a list of tasks by their order in the taskmap.
    """
    return sorted(tasklist, key=lambda task: taskmap[task].info.order)


def populate_dependents(taskmap: SimulatedTaskMap):
    """
    Populate the dependents field of each task.
    @param taskmap: SimulatedTaskMap
    """
    for task in taskmap.values():
        for dependency in task.dependencies:
            taskmap[dependency].dependents.append(task.name)


def apply_mapping(taskmap: SimulatedTaskMap, device: Optional[Device]):
    """
    Apply a mapping to a taskmap.
    @param taskmap: SimulatedTaskMap
    @param device: Device
    """
    for task in taskmap.values():
        task.info.mapping = device


def _compute_depth_internal(taskmap: SimulatedTaskMap, task: SimulatedTask, depth: int):
    task.depth = max(depth, task.depth)
    for dependent in task.dependents:
        _compute_depth_internal(taskmap, taskmap[dependent], depth + 1)


def _compute_depth(taskmap: SimulatedTaskMap, heads: Sequence[SimulatedTask]):
    for head in heads:
        _compute_depth_internal(taskmap, head, 0)


def compute_depths(taskmap: SimulatedTaskMap):
    """
    The function "compute_depths" calculates the depth of each task in a task map.

    :param taskmap: The `taskmap` parameter is a dictionary that represents a simulated task map.
    The `Task` class has a `dependencies` attribute, which is a list of task IDs that the task depends on.
    :type taskmap: SimulatedTaskMap
    """
    heads = [task for task in taskmap.values() if len(task.dependencies) == 0]
    _compute_depth(taskmap, heads)


def get_initial_tasks(taskmap: SimulatedTaskMap) -> List[TaskID]:
    """
    Get a list of tasks that have no dependencies.
    """
    return [task.name for task in taskmap.values() if len(task.dependencies) == 0]


def get_terminal_tasks(taskmap: SimulatedTaskMap) -> List[TaskID]:
    """
    Get a list of tasks that have no dependents.
    """
    return [task.name for task in taskmap.values() if len(task.dependents) == 0]
