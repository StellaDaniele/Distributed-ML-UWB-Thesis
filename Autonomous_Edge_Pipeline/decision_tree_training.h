#ifndef DECISION_TREE_TRAINING_H

#define DECISION_TREE_TRAINING_H

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include "main.h"
#include "dataset.h"
#include "kmeans.h"
#include "pipeline.h"

#define MAX_DEPTH 3
#define MIN_SIZE 10

struct Node{
	float threshold;
    int feature;
	int Left_group[MEMORY_SIZE];
	int Right_group[MEMORY_SIZE];
	float left_counter;
	float right_counter;
	int left_class;
	int right_class;
	struct Node* left;
	struct Node* right;
	};


struct Node* decision_tree_training(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR], int size);
struct Node* get_split(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int* group, int y_train[MEMORY_SIZE+UPDATE_THR], int size);
struct Node* split_samples(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int* group, int feature, float threshold, int size);
float gini_index(struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR]);
struct Node* split(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* node, int y_train[MEMORY_SIZE+UPDATE_THR], int max_depth, int min_size, int depth);
struct Node* GetNewNode();
int to_terminal(int *group, int y_train[MEMORY_SIZE+UPDATE_THR], int size);
#endif