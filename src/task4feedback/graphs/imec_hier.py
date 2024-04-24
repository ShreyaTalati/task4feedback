#Blocked GEMM + Cannon GEMM
import math
from .utilities import *
from ..types import *

@dataclass(slots=True)
class ImecHierDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a reduction.
    """
    hier_levels: int = 2
    # data_size: int = 256 * 256 * 8 # 4 * (n**2/p) n=2**8 and p=64
    # n_devices: int = 4
    levels: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    energy: list = field(default_factory=list)
    num_gpus: list = field(default_factory=list) # [100, 8100]
    p_per_mesh: list = field(default_factory=list) # [100, 9]
    # energy: list[float] = [0.01, 1, 100, 1000]  
    # energy: float = 0.01
    n: int = 8
    a: int = 8
    p: int = 4
    total_n: int = 40

    def initial_data_size(self, data_id: DataID) -> Devices:
        hier_level = data_id.idx[0][2]
        idx = self.hier_levels - hier_level - 1
        num_proc = self.num_gpus[idx]
        n_on_each_proc = (self.n * self.n) / num_proc
        return n_on_each_proc * self.a
    
    def initial_data_placement(self, data_id: DataID) -> Devices:
        start_gpu = 0
        idx = self.hier_levels - data_id.idx[0][2] - 1
        mesh_number = data_id.idx[0][3]
        level = data_id.idx[0][4]
        j = data_id.idx[0][5]
        if(level == self.levels[idx] - 1):
            return Device(Architecture.GPU, self.total_n, 4.24) # each A & B block of highest level of Cannon gemm reads from HBM
        start_gpu = 0
        
        for i in range(idx - 1, -1, -1):
            start_gpu += self.num_gpus[i]
        pos = start_gpu + mesh_number * self.p_per_mesh[idx]
        if(level == 0):
            pos += j
        else:
            idx_pos = j % (2 * self.blocks[idx])
            if(j >= 2 * self.blocks[idx]):
                pos += int(idx_pos) # To accomodate C
            else:
                # pos += int(data_id.idx[0][3] // 2) #If only A,B read
                pos += int(idx_pos // 2) # To accomodate A & B
        #return Device(Architecture.GPU, pos)
        # print("Data id: ", data_id.idx, " ", pos)
        # print("test: data_place ", str(data_id.idx[0]), " ", pos, " ", energy[data_id.idx[0][0]])
        print("cannon: ", data_id.idx, " ", pos, " ", self.energy[data_id.idx[0][2]])
        return Device(Architecture.GPU, pos, self.energy[data_id.idx[0][2]])
        # return Device(Architecture.GPU, self.total_n, 4.24) # everyting is on HBM at the start


    def __post_init__(self):
        self.initial_placement = self.initial_data_placement
        self.initial_sizes = self.initial_data_size

        def edges(task_id: TaskID):
            #print("Task ID: ", task_id.task_idx)
            in_data_indices = []
            irow = task_id.task_idx[0]
            jcol = task_id.task_idx[1]
            k = task_id.task_idx[2]
            hier_level = task_id.task_idx[3]
            mesh_number = task_id.task_idx[4]
            level = task_id.task_idx[5]
            j = task_id.task_idx[6]
            idx = self.hier_levels - hier_level - 1
            step = int(math.sqrt(self.blocks[idx]))
            # mod = 2 * self.blocks - 1
            # print("TASK ID:", task_id.task_idx)
            start_row = (j // step) * step
            start_col = j % step
            shift = (self.levels[idx] - 1) - level
            mod_a = start_row + step
            mod_b = self.blocks[idx]
            # step = self.branch_factor ** (self.levels - level)
            # start = step * j
            if(level == 0):
                data_info = TaskDataInfo()
                return data_info

            elif(level == self.levels[idx] - 1): # read data at the topmost level
                in_data_indices.append((irow, k, hier_level, mesh_number, step + 1, 2 * j)) # read A block
                in_data_indices.append((k, jcol, hier_level, mesh_number, step + 1, 2 * j + 1)) # read B block
                in_data_indices.append((irow, jcol, hier_level, mesh_number, step + 1, 2 * self.blocks[idx] + j)) # read C block
                # print("IF in_data_indices: ", in_data_indices)
            elif(j == mod_a - 1):
                in_data_indices.append((irow, k, hier_level, mesh_number, step + 1, (2 * ((start_col + shift) % step + start_row)))) # read A block
                # in_data_indices.append((step + 1, (2 * ((j + shift) % mod_a + start_row)))) # read A block
                in_data_indices.append((k, jcol, hier_level, mesh_number, step + 1, (2 * ((j + step * shift) % mod_b) + 1))) # read B block
                # in_data_indices.append((step + 1, ((2 * ((j + step * shift) + 1) % mod)))) # read B block
                # print("ELIF in_data_indices: ", in_data_indices, task_id.task_idx)
            else:
                # if(j == 14 or j == 15):
                    # print("start_col: ", start_col)
                    # print("shift: ", shift)
                    # print("mod_b: ", mod_b)
                    # print("start_row: ", start_row)
                in_data_indices.append((irow, k, hier_level, mesh_number, step + 1, (2 * ((start_col + shift) % step + start_row)))) # read A block
                # in_data_indices.append((step + 1, (2 * ((j + shift) % mod_a)))) # read A block
                in_data_indices.append((k, jcol, hier_level, mesh_number, step + 1, (2 * ((j + shift * step) % mod_b) + 1))) # read B block
                # in_data_indices.append((step + 1, ((2 * ((j + step * shift) + 1) % mod)))) # read B block
                # print("ELSE in_data_indices: ", in_data_indices, task_id.task_idx)
            #out_data_index = (hier_level, mesh_number, 0, j) # always write to addition
            # out_data_index = []
            # if(level == 1):
            #     out_data_index.append((hier_level, mesh_number, 0, j)) # write C at last

            #inout_data_index = start
            #print(in_data_indices)
            
            data_info = TaskDataInfo()
            for i in in_data_indices:
                #print(i)
                data_info.read_write.append(DataAccess(DataID((i,)), device=0))
            #print(out_data_index)
            # for i in out_data_index:
            #     print(task_id.task_idx, " ", out_data_index)
            #     data_info.write.append(DataAccess(DataID((i,)), device=0))

            return data_info

        self.edges = edges


def imec_task_mapping_gpu(
    task_id: TaskID, n_devices: int = 4, branch_factor: int = 2
) -> Devices:
    level = task_id.task_idx[0]
    j = task_id.task_idx[1]

    tasks_in_level = branch_factor ** (level)
    subtree_segment = tasks_in_level / n_devices
    device_index = int(j // subtree_segment)
    return Device(Architecture.GPU, device_id=device_index)

@dataclass(slots=True)
class ImecHierConfig(GraphConfig):
    levels: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    hier_levels: int = 2
    num_gpus: list = field(default_factory=list) # [100, 8100]
    n: int = 180
    #levels: int = 3
    #blocks: int = 4
    num_meshes: list = field(default_factory=list) # [1, 10]
    a: int = 8
    B: int = 2


@register_graph_generator
def make_imec_hier_graph(
    config: ImecHierConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    prev = []

    # return task_dict, data_dict
    for irow in range(config.B):
        for jcol in range(config.B):
            for k in range(config.B):
                for i in range(config.hier_levels - 1, -1, -1):
                    idx = config.hier_levels - i - 1
                    num_proc = config.num_gpus[idx]
                    n_on_each_proc = (config.n * config.n) / num_proc
                    data_size = config.a * n_on_each_proc
                    num_meshes = config.num_meshes[idx]
                    for mesh in range(num_meshes):
                        for level in range(config.levels[idx] - 1, -1, -1): #levels are going to be sq_root of # blocks + 1
                            tasks_in_level = config.blocks[idx]
                            # subtree_segment = tasks_in_level / config.n_devices

                            for j in range(tasks_in_level):
                                # Task ID:
                                #print(j)
                                task_idx = (irow, jcol, k, i, mesh, level, j)
                                task_id = TaskID("T", task_idx, 0)
                                # print("Task Id: ", task_id)

                                # Task Placement Info
                                task_placement_info = configurations(task_id)

                                # Task Dependencies
                                dependency_list = []

                                # Add cannon gemm depenedencies present within a mesh
                                if level == 0: # addition depends on all the prior multiplication
                                    for l in range(config.levels[idx] - 1):
                                        # print((level + k + 1, j))
                                        dependency = TaskID(
                                            "T", (irow, jcol, k, i, mesh, level + l + 1, j), 0
                                        )
                                        # print(dependency)
                                        dependency_list.append(dependency)
                                        prev.append(task_id) # add completion tasks of this cannon gemm as dependency of next cannon gemm
                                    # print(dependency_list)

                                elif level < config.levels[idx] - 1: # all multiplications except 1 can take place only after all prior level tasks are finished
                                    for dep in range(tasks_in_level):
                                        dependency = TaskID(
                                            "T", (irow, jcol, k, i, mesh, level + 1, dep), 0
                                        )
                                        dependency_list.append(dependency)
                                        #for k in range(config.branch_factor):
                    
                                # prev_level_mesh = int(pow(config.p, config.hier_levels - i)) #number of mesh in the previous level
                    
                                # Add dependencies between different system levels
                                # Depends if the prev levels all tasks are completed for all meshes
                                # If level=0 task is completed, it means, computation of a mesh is complete 
                                elif level == config.levels[idx] - 1 and len(prev) != 0: #next Cannon Gemm depends on completion of previous cannon gemm
                                    for l in prev:
                                        dependency_list.append(l)
                                        prev = []
                                # if i > 0:
                                #     #print("prev_level_mesh: ", prev_level_mesh)
                                #     for l in range(prev_level_mesh):
                                #         for k in range(tasks_in_level):
                                #             dependency = TaskID(
                                #                 "T", (i - 1, l, 0, k), 0
                                #             )
                                #         # print(dependency)
                                #         dependency_list.append(dependency)

                                    # print(dependency_list)
                                    # print("level: ", (level,j))
                                data_dependencies, data_dict = get_data_dependencies(
                                    task_id, data_dict, data_config
                                )

                                # Task Mapping
                                task_mapping = get_mapping(config, task_id)
                                # if(task_idx[0] == 0):
                                #     print("taskId: " + str(task_idx) + " " + str(task_placement_info))
                                task_dict[task_id] = TaskInfo(
                                    task_id,
                                    task_placement_info,
                                    dependency_list,
                                    data_dependencies,
                                    task_mapping,
                                )

    return task_dict, data_dict

