#include <stdio.h>
#include <string.h>
#include "main.h"
#include "kmeans.h"
#include "dataset.h"
// #include "decision_tree_training.h"
#include "pipeline.h"
#include "test.h"
#include "KNN_classification.h"
// #include "decision_tree_classification.h"
#include "time.h"

float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE];
int y_train[MEMORY_SIZE+UPDATE_THR];
float centroids[K][N_FEATURE];
float weights[MEMORY_SIZE+UPDATE_THR][K];

char* log_file_name = "log_"SETTINGS".txt";

int main()
{
    int n_samples;
    int increment = 0;
    float acc = 0;
    float acc_perm = 0;
    int counter = 0;
    int pred_class, pred_class_perm;

    #ifdef ONE_SHOT
    n_samples = MEMORY_SIZE;
    /* READ ONE_SHOT DATA */
    for(int i = 0; i < n_samples; i++)
    {
        for(int j = 0; j < N_FEATURE; j++)
        {
            max_samples[i][j] = X_train[i+NODE_OFFSET][j];
        }
    }

    #else
    /* MAX MEMORY ALLOCATION */
    n_samples = INITIAL_THR;
    /* READ INITIAL DATA */
    for(int i = 0; i < n_samples; i++)
    {
        for(int j = 0; j < N_FEATURE; j++)
        {
            max_samples[i][j] = X_train[i+NODE_OFFSET][j];
        }
    }
    #endif

    /* Save info into log file */
    FILE *fptr;
    fptr = fopen(log_file_name, "w");
    fprintf (fptr, "Training set size: %d\n", MEMORY_SIZE);
    fprintf (fptr, "Testinig set size: %d\n\n", N_TEST);
    fprintf(fptr, "* k-means clustering:\n\n");
    fprintf(fptr, "\t- Number of clusters: %d\n", K);
    fprintf(fptr, "\t- Maximum number of iterations: %d\n\n", ITERATION);

    #ifdef AutoDT
    fprintf(fptr, "* Decision Tree classifier: \n\n");
    fprintf(fptr, "\t- Max Depth: %d\n", MAX_DEPTH);
    fprintf(fptr, "\t- Min Size: %d\n\n", MIN_SIZE);
    #endif

    #ifdef AutoKNN
    fprintf(fptr, "* KNN classifier:\n\n");
    fprintf(fptr, "\t- Number of neighbors: %d\n\n", K_NEIGHBOR);
    #endif

    fprintf (fptr, "* Pipeline:\n\n");
    fprintf (fptr, "\t- Memory size: %d\n", MEMORY_SIZE);
    fprintf (fptr, "\t- Initial threshold size: %d\n", INITIAL_THR);
    fprintf (fptr, "\t- Update threshold: %d\n", UPDATE_THR);
    fprintf(fptr, "\t- Filtering strategy: %s\n\n", FILTER);
    fclose(fptr);

    /*
    counter to know how much samples I need before going to pipeline because we have limited number
    of samples in the dataset (different than a real reading from sensors scenario)
    */
    counter = n_samples;

    while (1)
    {
        n_samples = kmeans(max_samples, centroids, weights, y_train, n_samples);

        if(n_samples > MEMORY_SIZE)
        {
            #ifdef CONF
            int indices[MEMORY_SIZE + UPDATE_THR];

            for(int i=0; i<n_samples; i++)
            {
                indices[i]=i;
            }

            quicksort_idx(y_train, indices, 0, n_samples-1);
            n_samples = update_mem(max_samples, indices, n_samples);
            #endif

            #ifdef FIFO
            for(int i = 0; i < MEMORY_SIZE; i++)
            {
                for(int j = 0; j < N_FEATURE; j++)
                {
                    max_samples[i][j] = max_samples[i+(n_samples - MEMORY_SIZE)][j];
                }
                y_train[i] = y_train[i+(n_samples - MEMORY_SIZE)];
            }
            n_samples = MEMORY_SIZE;
            #endif

            #ifdef RANDOM
            int idx_to_replace[UPDATE_THR];
            random_func(idx_to_replace);
            for(int i = 0; i < (n_samples - MEMORY_SIZE); i++)
            {
                for(int j = 0; j < N_FEATURE; j++)
                {
                    max_samples[idx_to_replace[i]][j] = max_samples[MEMORY_SIZE + i][j];
                }
                y_train[idx_to_replace[i]] = y_train[MEMORY_SIZE+i];
            }
            n_samples = MEMORY_SIZE;
            #endif
        }
        #ifdef AutoDT
        struct Node* root = (struct Node*)realloc(NULL, sizeof(struct Node));

        decision_tree_training(max_samples, root, y_train, n_samples);
        #endif

        for(int j = 0; j < N_TEST; j++)
        {
            #ifdef AutoKNN
            pred_class = knn_classification(X_test[j], max_samples, y_train, n_samples);
            #endif

            #ifdef AutoDT
            pred_class = decision_tree_classifier(root, X_test[j]);
            #endif

            pred_class_perm = 1 - pred_class;

            if(pred_class == y_test[j])
            {
                acc++;
            }
            else if(pred_class_perm == y_test[j])
            {
                acc_perm++;
            }
        }
        if (acc_perm > acc)
        {
            acc = acc_perm;
        }

        fptr = fopen(log_file_name, "a");
        #ifdef AutoDT
        fprintf (fptr, "^ Decision Tree:\n\n");
        fprintf (fptr, "\t- Number of samples correctly classified using the Decision Tree: %0.0f\n", acc);
        #endif

        #ifdef AutoKNN
        fprintf(fptr, "^ KNN: \n\n");
        fprintf (fptr, "\t- Number of samples correctly classified using the KNN classifier: %0.0f\n", acc);
        #endif
        acc = (acc/N_TEST) * 100;
        fprintf (fptr, "\t- Accuracy: %0.2f%s\n\n", acc, "%");
        fclose(fptr);

        #ifdef ONE_SHOT
        break;
        #endif

        counter = counter + UPDATE_THR;
        acc = 0;
        acc_perm = 0;

        if(counter > N_TRAIN)
        {
            break;
        }
        else
        {
            // n_samples = pipeline(max_samples, root, y_train, n_samples, counter);
            n_samples = pipeline(max_samples, y_train, n_samples, counter);
        }

        if(counter - INITIAL_THR == MEMORY_SIZE)
        {
            increment = INITIAL_THR;
        }
        else if(counter > MEMORY_SIZE)
        {
            increment += UPDATE_THR;
        }

    }
}


void quicksort_idx(int y_train[MEMORY_SIZE+UPDATE_THR], int indices[MEMORY_SIZE + UPDATE_THR], int first, int last){
   int i, j, pivot, temp;

   if(first>=MEMORY_SIZE){
       return;
   }// Avoid useless computation, as the other samples will be cut

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(weights[indices[i]][y_train[indices[i]]]>=weights[indices[pivot]][y_train[indices[pivot]]]&&i<last)
            i++;
         while(weights[indices[j]][y_train[indices[j]]]<weights[indices[pivot]][y_train[indices[pivot]]])
            j--;
         if(i<j){
            temp=indices[i];
            indices[i]=indices[j];
            indices[j]=temp;
         }
      }

        temp=indices[pivot];
        indices[pivot]=indices[j];
        indices[j]=temp;

      quicksort_idx(y_train, indices,first,j-1);
      quicksort_idx(y_train, indices,j+1,last);

   }
}


int update_mem(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], int indices[MEMORY_SIZE+UPDATE_THR],int n_samples){
    int n_samples_updated = n_samples;
    if (n_samples > MEMORY_SIZE) {
        n_samples_updated = MEMORY_SIZE;
    }
    int n_rows_erased=0;
    int n_indices_found=0;
    for(int i=0; i<n_samples; i++){
        bool row_to_keep =false;
        for(int j=0; j<n_samples_updated-n_indices_found; j++){
            if (i==indices[j]){
                row_to_keep=true;
                for(int k=j; k<n_samples_updated-n_indices_found-1; k++){
                    indices[k]=indices[k+1];
                }
                n_indices_found++;
                break;
            }
        }
        if (!row_to_keep){
            for(int j=i-n_rows_erased; j<n_samples-1-n_rows_erased; j++){
                for(int k=0; k<N_FEATURE; k++){
                    max_samples[j][k]=max_samples[j+1][k];
                    y_train[j] = y_train[j+1];
                }
            }
            n_rows_erased++;
        }
    }
    n_samples = n_samples_updated;
    return n_samples;
}


int* random_func(int idx_to_replace[])
{
    /* The algorithm works as follows: iterate through all numbers from 1 to N and select the
    * current number with probability rm / rn, where rm is how many numbers we still need to find,
    * and rn is how many numbers we still need to iterate through */
    int in, im;
    im = 0;
    time_t t;
    srand((unsigned) time(&t));

    for (in = 0; in < MEMORY_SIZE && im < UPDATE_THR; ++in)
    {
        int rn = MEMORY_SIZE - in;
        int rm = UPDATE_THR - im;
        if (rand() % rn < rm)
        {
            /* Take it */
            idx_to_replace[im++] = in;
        }
    }
    return idx_to_replace;
}
