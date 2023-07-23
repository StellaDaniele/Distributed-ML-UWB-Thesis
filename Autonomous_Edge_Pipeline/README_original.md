# AEP
We introduce the Self-Learning Autonomous Edge Learning and Inferencing Pipeline (AEP). AEP is a software system we have developed for resource-constrained devices. Beside the inference capability, the AEP allows autonomous field training, by means of a two stage pipeline, involving a label generation with a confidence measure step and on-device training.

For label generation, AEP implements the following clustering algorithm:

- `k-means`: 

For on-device training, AEP currently implements the following ML algorithms (training and classification) for binary classification problems:

- `Decision Tree`
- `K-NN`

## Usage

You can compile the code as an executable or as a static library, using gcc/g++ for a Microcontroller or a desktop (e.g., thorugh Eclipse CDT or Visual Studio Code).

The program must be configured first in `main.h`, where the user has to specify some `#define`, such as:
- **Algorithm**: to enable one of the two supported classifirers
  * AutoDT: the DT training algorithm is executed,
  * AutoKNN: the samples together with their corresponding pseudo-labels (resulting from k-means clustering) are ready to be used by the K-NN classifier,

- **Confidence**: to enable/disable the confidence algorithm
  * CONFIDENCE: the k-means labels are evaluated by a confidence algorithm which filters the samples to be used for training one of the two supported classifiers (can be enabled/disabled),

- **Filter**: to enable one of the three supported memory filtering strategies
  * FIFO: removes the older samples from memory,
  * RND: removes random samples from memory, 
  * CONF: removes samples having low confidence values, 

Second, the memory should be configured in `pipeline.h`, where the user has to choose values for:
  * INITIAL_THR: initial number of samples to start the pipeline,
  * UPDATE_THR: after this number of samples, the k-means clustering is performed again for all samples in the memory, providing the new set of pseudo-labels and updating the classifiers accordingly,
  * MEMORY_SIZE: maximum number of samples that fit in memory,

Third, the k-means algorithm should be configured in `kmeans.h`, where the user has to choose values for:
  * K: number of clusters,
  * ITERATION: maximum number of iterations of the k-means algorithm for a single run.
  * CONFIDENCE_THR: the threshold for the confidence algorithm, so to remove all samples having weights lower than the threshold

#### AutoDT
 If `AutoDT` was set, the program must be configured in `decision_tree_training.h`, where the user has to choose values for:
   * MAX_DEPTH: maximum depth of the tree,
   * MIN_SIZE: minimum number of samples required to split an internal node,

#### AutoKNN
 If `AutoKNN` was set, the program must be configured in `knn_classification.h`, where the user has to choose value for:
   * K_NEIGHBOR: number of neighbors k to be considered in each decision,

`main.h` exposes the following functions:
- *`kmeans`*, k-means clustering algorithm. More details about this function can be found in the `kmeans.c` file
- *`decision_tree_training`*, the decision tree training algorithm. More details about this function can be found in the `decision_tree_training.c` file
- *`decision_tree_classifier`*, the decision tree classification algorithm. More details about this function can be found in the `decision_tree_classifier.c` file
- *`knn_classification`*, the K-NN classification algorithm. More details about this function can be found in the `knn_classification.c` file
kmeans_classifier
- *`pipeline`*, the pipeline function followed after clustering and training. More details about this function can be found in the `pipeline.c` file
- *`quicksort_idx` and `update_mem`*, these two functions are responsible of sorting the samples in memory to keep the most confident ones when `CONF` filtering is enables.
- *`random_func`*, this function is responsible of choosing the random samples to be removed when `RND` filtering is enables.

## Data
* float 32 data are used
* The dataset is saved as header and source files (dataset and test) for the training and testing phases respectively to evaluate the AEP and mimic a "reading sensory data" scenario (the data in this repository represent 80% trainng data and 20% testing data from the Pima Indian Dataset)

## Results
The obtained results after running the AEP with the user defined parameters are saved in a text file named *`log.txt`*

## Reference article for more infomation
F., Sakr, R. Berta, J. Doyle, A. De Gloria, and F., Bellotti, "Self-Learning Pipeline for Low-Energy Resource-Constrained Devices," Energies 2021, 14, 6636. https://www.mdpi.com/1996-1073/14/20/6636
