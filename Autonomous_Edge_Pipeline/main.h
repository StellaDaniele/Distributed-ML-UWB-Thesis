#ifndef MAIN_H
#define MAIN_H

// #include "decision_tree_training.h"
#include "pipeline.h"
#include "kmeans.h"
#include "KNN_classification.h"
#include "dataset.h"
#include "test.h"

// How likely each data point belong to each cluster.
#define CONFIDENCE

#ifndef SIMULATION
/* one-shot Implementation*/
#define ONE_SHOT

// #define AutoDT
#define AutoKNN

/* FILTERING STRATEGY */

#define CONF
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

#define NODE_OFFSET 100
#define SETTINGS "log"
#endif

extern char* log_file_name;

int kmeans(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], float weights[MEMORY_SIZE][K], int *y_train, int max);
// struct Node* decision_tree_training(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR], int size);
// int decision_tree_classifier(struct Node* root, float X[]);
int knn_classification(float X[], float training_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples);
// int pipeline(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, int counter);
int pipeline(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, int counter);
void quicksort_idx(int y_train[MEMORY_SIZE+UPDATE_THR], int indices[MEMORY_SIZE + UPDATE_THR], int first, int last);
int update_mem(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int indices[MEMORY_SIZE+UPDATE_THR],int n_samples);
int* random_func(int idx_to_replace[]);


#endif
