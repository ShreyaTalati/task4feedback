from task4feedback.graphs import *
from task4feedback.load import *

from rich import print

# from utility.execute import run
from task4feedback.visualize import *

from task4feedback.simulator.preprocess import *
from task4feedback.simulator.simulator import *
from task4feedback.simulator.topology import *

from task4feedback.simulator.analysis.recorder import *
from task4feedback.simulator.analysis.plot import *
from task4feedback.simulator.analysis.export import *
from task4feedback.simulator.interface import *
from task4feedback.simulator.verify import *

from time import perf_counter as clock


def test_data():
    cpu = Device(Architecture.CPU, 0)
    gpu0 = Device(Architecture.GPU, 0)
    gpu1 = Device(Architecture.GPU, 1)

    def initial_data_placement(data_id: DataID) -> Devices:
        return Device(Architecture.CPU, 0)

    def sizes(data_id: DataID) -> int:
        return 32 * 1024 * 1024  # 1 GB

    def task_duration_per_func(task_id: TaskID):
        duration = 4000
        if task_id.taskspace == "POTRF":
            duration = 8000
        elif task_id.taskspace == "SYRK":
            duration = 5000
        elif task_id.taskspace == "SOLVE":
            duration = 3000
        elif task_id.taskspace == "GEMM":
            duration = 3000
        return duration

    def func_type_id(task_id: TaskID):
        func_id = 0
        if task_id.taskspace == "POTRF":
            func_id = 0
        elif task_id.taskspace == "SYRK":
            func_id = 1
        elif task_id.taskspace == "SOLVE":
            func_id = 2
        elif task_id.taskspace == "GEMM":
            func_id = 3
        print(task_id.taskspace, " type id ", func_id)
        return func_id

    def task_placement(task_id: TaskID) -> TaskPlacementInfo:
        device_tuple = Device(Architecture.GPU, task_id.task_idx[0] % 4)

        runtime_info = TaskRuntimeInfo(
            task_time=task_duration_per_func(task_id), device_fraction=1,
            memory=int(0))
        placement_info = TaskPlacementInfo()
        placement_info.add(device_tuple, runtime_info)

        return placement_info

    data_config = CholeskyDataGraphConfig()
    # data_config = NoDataGraphConfig()
    data_config.initial_placement = initial_data_placement
    data_config.initial_sizes = sizes

    config = CholeskyConfig(blocks=10, task_config=task_placement,
                            func_id=func_type_id)
    tasks, data = make_graph(config, data_config=data_config)

    topology = TopologyManager().generate("frontera", config=None)

    simulator_config = SimulatorConfig(
        topology=topology,
        tasks=tasks,
        data=data,
        scheduler_type="parla",
        randomizer=Randomizer(),
    )
    simulator = create_simulator(config=simulator_config)

    start_t = clock()
    simulated_time = simulator.run()
    end_t = clock()

    print(f"Time to Simulate: {end_t - start_t}")
    print(f"Simulated Time: {simulated_time}")

    # print(
    #     simulator.recorders.get(LaunchedResourceUsageListRecorder).vcu_usage[
    #         Device(Architecture.GPU, 0)
    #     ]
    # )

    # make_resource_plot(
    #     recorder=simulator.recorders,
    #     resource_type=ResourceType.MEMORY,
    #     phase=TaskState.LAUNCHED,
    # )

    # print("Tasks: ")
    # print(summarize_dependencies(simulator_config.simulated_tasks))

    # for task in simulator_config.simulated_tasks.values():
    #    print(f"{task.name} {task.times}")

    # verify_order(simulator_config.simulated_tasks)
    # verify_runtime_resources(simulator_config.simulated_tasks, simulator.devicemap)

    # print(simulator.recorders)

    # make_plot(
    #     simulator.recorders.recorders[0],
    #     simulator.recorders.recorders[1],
    #     simulator.recorders.recorders[2],
    #     data_ids=[DataID((4, 1))],
    # )

    # export_task_records(
    #     simulator.recorders.get(ComputeTaskRecorder),
    #     simulator.recorders.get(DataTaskRecorder),
    #     filename="task_records.json",
    # )


if __name__ == "__main__":
    test_data()
