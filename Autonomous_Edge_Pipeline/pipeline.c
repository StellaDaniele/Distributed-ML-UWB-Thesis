/* ----------------------------------------------------------------------
 * Project:      Autonomous Edge Pipeline
 * Title:        pipeline.c
 * Description:  run the pipeline (kmeans & KNN) with two KNN filtering strategy
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "main.h"
#include "kmeans.h"
#include "dataset.h"
#include "pipeline.h"
#include "KNN_classification.h"
// #include "decision_tree_training.h"
// #include "decision_tree_classification.h"

/**
 * @brief pipeline
 * @param[in]       max_samples         pointer to data samples
 * @param[in]       root                decision tree root node
 * @param[in]       y_train             pointer to labels resulting from clustering
 * @param[in]       n_samples           number of samples
 * @param[in]       counter             counter to keep track of the total number of samples at each step
 * @return     The function returns the new number of samples for the next iteration (n_samples)
 */


int pred_class;

// int pipeline(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, int counter)
int pipeline(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, int counter)
{
    for(int i = 0; i < UPDATE_THR; i++)
    {
        #ifdef AutoKNN
        pred_class = knn_classification(X_train[i + counter - UPDATE_THR], max_samples, y_train, n_samples);
        #endif

        #ifdef AutoDT
        pred_class = decision_tree_classifier(root, X_train[i + counter - UPDATE_THR]);
        #endif
    }

    for(int i = 0; i < UPDATE_THR; i++)
    {
        for(int j = 0; j < N_FEATURE; j++)
        {
            max_samples[n_samples + i][j] = X_train[i + counter - UPDATE_THR][j];
        }
    }
    n_samples += UPDATE_THR;
    return n_samples;
}

