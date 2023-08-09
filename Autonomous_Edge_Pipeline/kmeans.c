/* ----------------------------------------------------------------------
 * Project:      Autonomous Edge Pipeline
 * Title:        kmeans.c
 * Description:  kmeans clustering algorithm
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

/**
 * @brief kmeans
 * @param[in]       max_samples         pointer to data samples
 * @param[in]       centroids           pointer to centroids
 * @param[in, out]  y_train             pointer to clustering labels
 * @param[in]       weights             pointer to the confidence weights
 * @param[in]       n_samples           number of samples
 * The function fills y_train with the labels resulting from clustering
 * and return the new number of samples (changed due to confidence that removes unwanted samples)
 */

/**
 * @brief initial_centroids
 * @param[in]       max_samples          pointer to data samples
 * @param[in]       centroids            pointer to centroids
 * @param[in]       n_samples            number of samples
 * The function chooses one random sample as initial centroids
 */

/**
 * @brief kmeans_plus_plus
 * @param[in]       max_samples          pointer to data samples
 * @param[in]       centroids            pointer to centroids
 * @param[in]       n_samples            number of samples
 * @param[in]       next_centoird        the second random centoid using kmeans++ approach
 * The function chooses the other initial centroids using kmeans++ approach
 */

/**
 * @brief clustering
 * @param[in]       X                      pointer to input samples
 * @param[in]       centroids              pointer to centroids
 * @param[in]       weights                pointer to the confidence weights
 * @param[in, out]  samples_per_cluster    array that hold the number of samples belonging to each cluster
 * @param[in]       index                  index of input sample: used to write its corresponding weight
 * @return     The function returns the clustering label
 */

/**
 * @brief update_cluster_assignment
 * @param[in]       max_samples            pointer to data samples
 * @param[in]       index                  cluster_assignment index
 * The function updates cluster_assignment with the sum of samples in each cluster
 */

/**
 * @brief update_centroids
 * @param[in]       centroids              pointer to centroids
 * @param[in]       samples_per_cluster    array that holds the number of samples belonging to each cluster
 * The function updates centroids by calculating the mean of the samples inside the same cluster
 */

float cluster_assignment[K][N_FEATURE];
float prev_centroids[K][N_FEATURE];
int is_equal;
int stop = 0;

int kmeans(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], float weights[MEMORY_SIZE][K], int y_train[MEMORY_SIZE+UPDATE_THR], int n_samples)
{
    int cluster;
    float weight;
    FILE *fptr;
    int counter = 0;
    initial_centroids(max_samples, centroids, n_samples);

    /* RUN KMEANS FOR ITERATION TIMES UNTIL NO FURTHER CHANGE IN RESULTS*/
    for (int iteration = 0; iteration < ITERATION; iteration++)
    {
        int samples_per_cluster[K] = {0};
        for (int j = 0; j < n_samples; j++)
        {
            cluster = clustering(max_samples[j], centroids, weights, samples_per_cluster, j);
            y_train[j] = cluster;
            update_cluster_assignment(max_samples[j], cluster);
        }

        /* Copy centroids elements to centroids to compare */
        for(int i = 0; i < K; i++)
        {
            for(int k = 0; k < N_FEATURE; k++)
            {
                prev_centroids[i][k] = centroids[i][k];
            }
        }
        update_centroids(centroids, samples_per_cluster);

        /* Stop the algorithm when the centroids of the newly formed clusters stop changing (centroids = prev_centroids).
        We made an update to the stoping criterion by checking twice that centroids are equal to prev_centroids. */
        is_equal = memcmp(centroids, prev_centroids, sizeof(prev_centroids));
        if(is_equal == 0)
        {
            stop++;
            if(stop == 2)
            {
                fptr = fopen(log_file_name, "a");
                fprintf(fptr, "^ k-means: \n\n");
                fprintf(fptr, "\t- Centroids stopped changing at iteration: %d\n", iteration);
                fclose(fptr);
                break;
            }
        }

        /* AFTER EACH ITERATION RESET cluster_assignment TO ZERO */
        memset(cluster_assignment, 0, sizeof(cluster_assignment));
    }

    #ifdef CONFIDENCE
    /* weight calculation */
    for(int i = 0; i < n_samples; i++)
    {
        weight = 0;
        for(int j = 0; j < K; j++)
        {
            weight += pow((1/weights[i][j]), 2);
        }
        for(int k = 0; k < K; k++)
        {
            weights[i][k] = 1 / (weight * pow(weights[i][k], 2));
        }
    }

    for(int n = 0; n < n_samples; n++)
    {
        if(weights[n][y_train[n]] > CONFIDENCE_THR)
        {
            for(int l = 0; l < N_FEATURE; l++)
            {
                max_samples[n-counter][l] = max_samples[n][l];
            }
            y_train[n-counter] = y_train[n];
        }
        else
        {
            counter++;
        }
    }
    #endif
    n_samples = n_samples - counter;

    // for(int n = 0; n < n_samples; n++)
    // {
    //     printf("samples[%d][0]: %f\n", n, samples[n][0]);
    // }

    fptr = fopen(log_file_name, "a");
    fprintf(fptr, "^ k-means: \n\n");
    fprintf(fptr, "\t- Final Centroids are: \n");
    for(int i = 0; i < K; i++)
    {
        fprintf(fptr, "\t[\t");
        for(int j = 0; j < N_FEATURE; j++)
        {
            fprintf(fptr, "%f\t", centroids[i][j]);
        }
        fprintf(fptr, "]\n");
    }
    fprintf(fptr, "\n");
    fclose(fptr);
    return n_samples;
}

/* RANDOMLY CHOOSE ONE SAMPLE AS INITIAL CENTROIDS */
void initial_centroids(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], int n_samples)
{
    time_t t;
    srand((unsigned) time(&t));

    int random = rand() % n_samples;

    for(int i = 0; i < K; i++)
    {
        for(int j = 0; j < N_FEATURE; j++)
        {
            centroids[i][j] = max_samples[random][j];
        }
        if(i+1 == K)
        {
            break;
        }
        /* TO CHOOSE THE OTHER CENTROIDS USE KMEANS++*/
        random = kmeans_plus_plus(max_samples, centroids, n_samples, i+1);
    }
}

int kmeans_plus_plus(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], float centroids[K][N_FEATURE], int n_samples, int next_centroid)
{
    float max = -1000;
    int random;
    float dist;

    for (int i = 0; i < n_samples; i++)
    {
        for (int k = 0; k < next_centroid; k++)
        {
            for (int j = 0; j < N_FEATURE; j++)
            {
                dist += pow((max_samples[i][j] - centroids[k][j]), 2.0);
            }

            if (dist > max)
            {
                max = dist;
                random = i;
            }
            dist = 0;
        }
    }
    return random;
}

/* EACH DATA POINT IS ALLOCATED TO THE NEAREST CENTROID */
int clustering(float X[], float centroids[K][N_FEATURE], float weights[MEMORY_SIZE][K], int samples_per_cluster[], int index)
{
    float y = 0;
    float min_distance = 1000000;
    int cluster = 0;

    for (int k = 0; k < K; k++)
    {
        for (int j = 0; j < N_FEATURE; j++)
        {
            y += pow((X[j] - centroids[k][j]), 2.0);
        }
        /* FUZZY CLUSTERING */
        weights[index][k] = y;

        y = sqrt(y);

        if (y < min_distance)
        {
            min_distance = y;
            y = 0;
            cluster = k;
            continue;
        }
        y = 0;
    }
    samples_per_cluster[cluster] += 1;
    return cluster;
}

/* UPDATE cluster_assignment --> SUM OF THE SAMPLES ASSIGNED TO THE SAME CLUSTER */
void update_cluster_assignment(float max_samples[MEMORY_SIZE+UPDATE_THR], int index)
{
    for (int i = 0; i < N_FEATURE; i++)
    {
        cluster_assignment[index][i] = cluster_assignment[index][i] + max_samples[i];
    }
}

/* UPDATE centroids --> MEAN OF THE SAMPLES INSIDE THE SAME CLUSTER */
void update_centroids(float centroids[K][N_FEATURE], int samples_per_cluster[])
{
    for (int j = 0; j < N_FEATURE; j++)
    {
        if(samples_per_cluster[0] != 0)
        {
            centroids[0][j] = (cluster_assignment[0][j])/(samples_per_cluster[0]);
        }
        if(samples_per_cluster[1] != 0)
        {
            centroids[1][j] = (cluster_assignment[1][j])/(samples_per_cluster[1]);
        }
    }
}
