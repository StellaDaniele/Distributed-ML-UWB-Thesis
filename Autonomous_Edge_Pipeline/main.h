#ifndef MAIN_H
#define MAIN_H

// #include "decision_tree_training.h"
#include "pipeline.h"
#include "kmeans.h"
#include "KNN_classification.h"
#include "dataset.h"
#include "test.h"

// #define AutoDT
#define AutoKNN

#ifndef SIMULATION
/* one-shot Implementation*/
#define ONE_SHOT 0
// How likely each data point belong to each cluster.
#define CONFIDENCE 1
/* FILTERING STRATEGY */
#define CONF
#define NODE_ID 0
#define N_NODES 1
#define NODE_OFFSET 0
#define N_TRAIN_USED N_TRAIN
#define SETTINGS "log"
#define N_TEST_USED N_TEST
#endif

#ifdef CONF
#define FILTER "CONF"
#endif

// #define FIFO
#ifdef FIFO
#define FILTER "FIFO"
#endif

// #define RANDOM
#ifdef RANDOM
#define FILTER "RANDOM"
#endif

extern char* log_file_name;

int kmeans(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], float weights[MEMORY_SIZE][K], int *y_train, int max);
// struct Node* decision_tree_training(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR], int size);
// int decision_tree_classifier(struct Node* root, float X[]);
//int knn_classification(float X[], float training_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, struct DataPoint nearest_neighbors[K_NEIGHBOR]);
// int pipeline(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, int counter);
int pipeline(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, int counter);
void quicksort_idx(int y_train[MEMORY_SIZE+UPDATE_THR], int indices[MEMORY_SIZE + UPDATE_THR], int first, int last);
int update_mem(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int indices[MEMORY_SIZE+UPDATE_THR],int n_samples);
int* random_func(int idx_to_replace[]);


#endif
