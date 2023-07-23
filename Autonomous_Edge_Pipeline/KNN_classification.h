#ifndef KNN_CLASSIFICATION_H

#define KNN_CLASSIFICATION_H

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include "main.h"
#include "dataset.h"
#include "kmeans.h"
#include "pipeline.h"

#define K_NEIGHBOR 5

struct neighbour
{
  int id;
  float score;
};

int knn_classification(float X[], float training_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples);
int struct_cmp_by_score_dec(const void *, const void *);

#endif