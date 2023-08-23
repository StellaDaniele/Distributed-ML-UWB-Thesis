/* ----------------------------------------------------------------------
 * Project:      Autonomous Edge Pipeline
 * Title:        KNN_classification.c
 * Description:  KNN classifier
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "KNN_classification.h"

/**
 * @brief knn_classification
 * @param[in]       X                   pointer to input sample
 * @param[in]       training_samples    pointer to training data samples
 * @param[in]       y_train             pointer to labels resulting from clustering
 * @param[in]       n_samples           number of samples
 * @return     The function returns the best class for the X sample
 */

int knn_classification(float X[], float training_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples, struct DataPoint nearest_neighbors[K_NEIGHBOR])
{
    struct neighbour neighbours[MEMORY_SIZE];
    int j;

    for(j=0; j < n_samples; j++)
    {
        neighbours[j].id = j;
        float acc=0;
        bool skip=false;
        int k;

        for(k=0; k < N_FEATURE; k++)
        {
            acc+=(X[k] - training_samples[j][k])*(X[k] - training_samples[j][k]);
            if (acc > 10000000000)
            {
                neighbours[j].score = 0;
                skip=true;
                break;
            }
        }
        if (!skip)
        {
            acc = sqrt(acc);
            if (acc < 0.00000001)
            {
                neighbours[j].score = 100000000;
            } else
            {
                neighbours[j].score = 1 / acc;
            }
        }
    }

    qsort(neighbours, n_samples, sizeof(struct neighbour), struct_cmp_by_score_dec);
    {
        // Store coordinates and scores of k nearest neighbors
        for (int n = 0; n < K_NEIGHBOR; n++)
        {
            int neighbor_id = neighbours[n].id;
            if(nearest_neighbors != NULL){
                nearest_neighbors[n].score = neighbours[n].score;
                nearest_neighbors[n].label = y_train[neighbor_id];
                memcpy(nearest_neighbors[n].coords, training_samples[neighbor_id], N_FEATURE * sizeof(float));
            }
        }

        int n;
        float scores[K];
        memset(scores, 0, K*sizeof(float));

        for(n = 0; n < K_NEIGHBOR; n++)
        {
            scores[y_train[neighbours[n].id]] += neighbours[n].score;
            //printf("%f\t",neighbours[n].score);
        }
        //puts("");

        float bestScore=0;
        int bestClass;

        for(n = 0; n < K; n++)
        {
            //printf("%f\t",scores[n]);
            if (scores[n] > bestScore)
            {
                bestScore = scores[n];
                bestClass = n;
            }
        }
        //puts("\n");
        return(bestClass);
    }
}

int struct_cmp_by_score_dec(const void *a, const void *b)
{
    struct neighbour *ia = (struct neighbour *)a;
    struct neighbour *ib = (struct neighbour *)b;
    return -(int)(100000.f*ia->score - 100000.f*ib->score);
}
