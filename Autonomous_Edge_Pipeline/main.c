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

char* log_file_name = OUTPUT_DIR"log_"SETTINGS".json";

int main()
{
    int n_samples;
    int increment = 0;
    float acc = 0;
    float acc_perm = 0;
    int counter = 0;
    int pred_class, pred_class_perm;

    #if ONE_SHOT
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
    fprintf(fptr, "{");
    fprintf(fptr, "\"N_NODES\":%d,", N_NODES);
    fprintf(fptr, "\"NODE_ID\":%d,", NODE_ID);
    fprintf(fptr, "\"K_NEIGHBOR\":%d,", K_NEIGHBOR);
    fprintf(fptr, "\"MEMORY_SIZE\":%d,", MEMORY_SIZE);
    fprintf(fptr, "\"NODE_OFFSET\":%d,", NODE_OFFSET);
    fprintf(fptr, "\"CONFIDENCE\":%s,",((CONFIDENCE == 1)?"true":"false"));
    fprintf(fptr, "\"CONFIDENCE_THR\":%f,", CONFIDENCE_THR);
    fprintf(fptr, "\"FILTER\":\"%s\",", FILTER);
    fprintf(fptr, "\"ONE_SHOT\":%s,",((ONE_SHOT == 1)?"true":"false"));
    fprintf(fptr, "\"INITIAL_THR\":%d,", INITIAL_THR);
    fprintf(fptr, "\"UPDATE_THR\":%d,", UPDATE_THR);
    fprintf(fptr, "\"K\":%d,", K);
    fprintf(fptr, "\"ITERATION\":%d,", ITERATION);
    fprintf(fptr, "\"N_TRAIN\":%d,", N_TRAIN);
    fprintf(fptr, "\"N_TRAIN_USED\":%d,", N_TRAIN_USED);
    fprintf(fptr, "\"N_TEST\":%d,", N_TEST);
    fprintf(fptr, "\"N_TEST_USED\":%d,", N_TEST_USED);


    #ifdef AutoDT
    fprintf(fptr, "* Decision Tree classifier: \n\n");
    fprintf(fptr, "\t- Max Depth: %d\n", MAX_DEPTH);
    fprintf(fptr, "\t- Min Size: %d\n\n", MIN_SIZE);
    #endif

    #ifdef AutoKNN
    //fprintf(fptr, "* KNN classifier:\n\n");

    #endif




    /*
    counter to know how much samples I need before going to pipeline because we have limited number
    of samples in the dataset (different than a real reading from sensors scenario)
    */
    counter = n_samples;
    fprintf(fptr, "\"pipeline_iterations\":[");
    fprintf(fptr, "{");
    fclose(fptr);
    bool need_for_comma = false;
    while (1)
    {
        fptr = fopen(log_file_name, "a");
        if(need_for_comma)
            fprintf(fptr, ",{");
        need_for_comma = true;
        fclose(fptr);
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

        struct DataPoint datapoint[K_NEIGHBOR];
        fptr = fopen(log_file_name, "a");

        fprintf (fptr, "\"test_data\":[");
        for(int j = 0; j < N_TEST_USED; j++)
        {
            #ifdef AutoKNN
            pred_class = knn_classification(X_test[j], max_samples, y_train, n_samples, datapoint);
            #endif
            fprintf(fptr, "{\"test_id\":%d,",j);
            fprintf(fptr, "\"test_coordinates\":[");
            fprintf(fptr, "%f", X_test[j][0]);
            for(int i = 1; i < N_FEATURE; i++){
                fprintf(fptr, ",%f", X_test[j][i]);
            }
            fprintf(fptr, "]");
            fprintf(fptr, ",");
            fprintf(fptr, "\"neighbors\":[");
            for(int i = 0; i < K_NEIGHBOR; i++){
                fprintf(fptr, "{\"score\":%f,\"label\":%d,\"coordinates\":[",datapoint[i].score, datapoint[i].label);
                fprintf(fptr, "%f",datapoint[i].coords[0]);
                for(int ii = 1; ii < N_FEATURE; ii++){
                    fprintf(fptr, ",%f",datapoint[i].coords[ii]);
                }
                fprintf(fptr, "]}");
                if(i != K_NEIGHBOR - 1)
                    fprintf(fptr, ",");
            }
            fprintf(fptr, "]}");
            if(j != N_TEST_USED - 1)
                fprintf(fptr, ",");
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
        fprintf(fptr, "],");
        bool centroids_swapped = false;
        if (acc_perm > acc)
        {
            acc = acc_perm;
            centroids_swapped = true;
        }

        fprintf(fptr, "\"CENTROIDS_SWAPPED\":%s,",((centroids_swapped == 1)?"true":"false"));

        #ifdef AutoDT
        fprintf (fptr, "^ Decision Tree:\n\n");
        fprintf (fptr, "\t- Number of samples correctly classified using the Decision Tree: %0.0f\n", acc);
        #endif

        #ifdef AutoKNN
        //fprintf(fptr, "^ KNN: \n\n");
        fprintf(fptr, "\"correctly_classified_samples\":%0.0f,", acc);
        #endif
        acc = (acc/N_TEST_USED) * 100;
        fprintf(fptr, "\"accuracy\":%0.2f", acc);
        fprintf(fptr, "}");
        fclose(fptr);

        #if ONE_SHOT
        break;
        #endif

        counter = counter + UPDATE_THR;
        acc = 0;
        acc_perm = 0;

        if(counter > N_TRAIN_USED)
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
    fptr = fopen(log_file_name, "a");
    fprintf(fptr, "]}");
    fclose(fptr);
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
