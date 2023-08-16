import os

# Paths and compilation
file_compilation_strings = "compilation_strings.bat"
directory_executables = "./Compiled"
source_code_files = "../Autonomous_Edge_Pipeline/*.c"
gcc_warning_suppression_options = "2>nul"

# Static values
K = 2   # Number of clusters k-means
ITERATION = 50 # Max number of iterations k-means
N_TRAIN = 614
N_TEST = 154
N_TEST_USED = N_TEST

def generate_compilation_strings():
    CONF_THR_LIST = [.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]
    FILTERS = ['CONF','FIFO','RANDOM']

    output_file_names = []

    output_file = open(file_compilation_strings, "w")

    for N_NODES in range(1,6):
        N_TRAIN_USED = N_TRAIN // N_NODES
        for NODE_ID in range(1,N_NODES+1):
            for K_NEIGHBOR in range(1,11):
                MEMORY_SIZE = 10 # Initial value to decide
                while MEMORY_SIZE <= N_TRAIN_USED:
                    min_initial_thr = min(MEMORY_SIZE//2,50)
                    max_initial_thr = MEMORY_SIZE // 2
                    NODE_OFFSET = MEMORY_SIZE/N_NODES*(NODE_ID-1)
                    for CONFIDENCE in range(2): # False / True
                        for CONFIDENCE_THR in (CONF_THR_LIST if CONFIDENCE else [0]):
                            # The condition above should limit all the useless cases
                            for FILTER in FILTERS:
                                for ONE_SHOT in range(2): # False / True
                                    for INITIAL_THR in (range(min_initial_thr,max_initial_thr,50) if not ONE_SHOT else [0]):
                                        # TODO: Decide how to increase: is min+50*i ok?
                                        update_thr_values = [0]
                                        if not ONE_SHOT:
                                            min_update_thr_value = 5
                                            max_update_thr_value = 100
                                            current_value = min_initial_thr
                                            while current_value <= max_update_thr_value and INITIAL_THR+current_value <=MEMORY_SIZE:
                                                update_thr_values.append(current_value)
                                                current_value *= 2
                                        for UPDATE_THR in update_thr_values:
                                            # TODO: Decide how to increase: is 5*2^i ok?
                                            # TODO: Decide order, add everything to string, remember SIMULATION macro
                                            name_curr = f"{N_NODES}_{NODE_ID}_{K_NEIGHBOR}_{MEMORY_SIZE}_{CONFIDENCE}_{CONFIDENCE_THR}_{FILTER}_{ONE_SHOT}_{INITIAL_THR}_{UPDATE_THR}_{K}_{ITERATION}_{N_TRAIN}_{N_TRAIN_USED}_{N_TEST}_{N_TEST_USED}"
                                            compilation_string = f"gcc -D SIMULATION "\
                                            f"-D N_NODES={N_NODES} "\
                                            f"-D NODE_ID={NODE_ID} "\
                                            f"-D K_NEIGHBOR={K_NEIGHBOR} "\
                                            f"-D MEMORY_SIZE={MEMORY_SIZE} "\
                                            f"-D CONFIDENCE={CONFIDENCE} "\
                                            f"-D CONFIDENCE_THR={CONFIDENCE_THR} "\
                                            f"-D FILTER={FILTER} "\
                                            f"-D ONE_SHOT={ONE_SHOT} "\
                                            f"-D INITIAL_THR={INITIAL_THR} "\
                                            f"-D UPDATE_THR={UPDATE_THR} "\
                                            f"-D K={K} "\
                                            f"-D ITERATION={ITERATION} "\
                                            f"-D N_TRAIN={N_TRAIN} "\
                                            f"-D N_TRAIN_USED={N_TRAIN_USED} "\
                                            f"-D N_TEST={N_TEST} "\
                                            f"-D N_TEST_USED={N_TEST_USED} "\
                                            f"-D SETTINGS=\"{name_curr}\" "\
                                            f"{gcc_warning_suppression_options} "\
                                            f"{source_code_files} "\
                                            f"-o {directory_executables}/{name_curr}"
                                            output_file.write(compilation_string + "\n")
                                            output_file_names.append(name_curr)
                    MEMORY_SIZE *= 2 # TODO: Decide how to increase: is 10*2^i ok?
    return output_file_names

if __name__ == "__main__":
    if not os.path.exists(directory_executables):
        os.makedirs(directory_executables)

    # Gcc compilation string generation
    output_file_names = generate_compilation_strings()
    for file_name in output_file_names:
        print(file_name)

    # # Gcc compilation execution
    # os.system(file_compilation_strings)

    # # Execution of all compiled programs
    # for executable_name in output_file_names:
    #     os.system(directory_executables + "/" + executable_name + ".exe")

    # Reading and concatenating all the logs
    # for log_name in output_file_names:
    #     log_file = open(directory_executables + "/" + "log_" + log_name + ".txt", "r")
