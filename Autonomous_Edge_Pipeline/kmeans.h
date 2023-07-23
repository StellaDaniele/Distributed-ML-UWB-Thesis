#ifndef KMEANS_H

#define KMEANS_H

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
// #include <iostream>
// #include <cstring>
#include <stdlib.h>
#include <string.h>

#include "main.h"
#include "dataset.h"
#include "pipeline.h"

#define K 2
#define ITERATION 50
#define CONFIDENCE_THR 0.9

void initial_centroids(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], int n_samples);
int clustering(float X[], float centroids[K][N_FEATURE], float weights[MEMORY_SIZE][K], int samples_per_cluster[], int index);
void update_cluster_assignment(float max_samples[MEMORY_SIZE+UPDATE_THR], int index);
void update_centroids(float centroids[K][N_FEATURE], int samples_per_cluster[]);
int kmeans(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], float weights[MEMORY_SIZE][K], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples);
int kmeans_plus_plus(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], int n_samples, int next_centroid);

#endif
