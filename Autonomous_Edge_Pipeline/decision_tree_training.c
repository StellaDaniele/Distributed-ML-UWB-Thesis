/* ----------------------------------------------------------------------
 * Project:      Autonomous Edge Pipeline
 * Title:        decision_tree_training.c
 * Description:  Decision Tree training algorithm
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
// #include <iostream>
// #include <cstring>
#include <stdlib.h>
#include <string.h>

#include "main.h"
#include "dataset.h"
#include "decision_tree_training.h"

/**
 * @brief decision_tree_training
 * @param[in]       max_samples         pointer to data samples
 * @param[in, out]  root                root node
 * @param[in, out]  y_train             pointer to clustering labels
 * @param[in, out]  size                number of samples
 * @return     The function returns the trained decision tree (root)
 */

/**
 * @brief split_samples
 * @param[in]       max_samples         pointer to data samples
 * @param[in, out]  root                root node
 * @param[in]		group             	pointer to child samples indexes (left/right) (if first split group = NULL, check code for more details)
 * @param[in]		feature             best feature for the current split
 * @param[in]		threshold           best threshold for the current split
 * @param[in]		size                number of samples in the current node
 * @return     The function returns the root node
 */

/**
 * @brief get_split
 * @param[in]       max_samples         pointer to data samples
 * @param[in, out]  root                root node
 * @param[in]		group             	pointer to child samples indexes (left/right)
 * @param[in]       y_train             pointer to clustering labels
 * @param[in]		size                number of samples in the current node
 * @return     The function returns the root node
 */

/**
 * @brief gini_index
 * @param[in]  		root                root node
 * @param[in]       y_train             pointer to clustering labels
 * @return     The function returns the gini value
 */

/**
 * @brief GetNewNode
 * @return     The function returns new empty child node
 */

/**
 * @brief split
 * @param[in]       max_samples         pointer to data samples
 * @param[in, out]  node                current node
 * @param[in]       y_train             pointer to clustering labels
 * @param[in]		max_depth           maximum tree depth
 * @param[in]		min_size            minimum number of samples required to split an internal node
 * @param[in]		depth            	depth at current node
 * @return     The function is recursive and returns the root node with all internal nodes
 */

/**
 * @brief to_terminal
 * @param[in]		group             	pointer to child samples indexes (left/right)
 * @param[in]       y_train             pointer to clustering labels
 * @param[in]		size                number of samples in the current node
 * @return     The function returns the leaf node class
 */

int counter = 1;
int n = 0;

struct Node* decision_tree_training(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int y_train[MEMORY_SIZE+UPDATE_THR], int size)
{
	get_split(max_samples, root, NULL, y_train, size);
	return root;
}

struct Node* split_samples(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int* group, int feature, float threshold, int size)
{
	int left_counter = 0;
	int right_counter = 0;
	int sample_index;

	for(int j = 0; j < size; j++)
	{
		/* first split of the samples */
		if(group == NULL)
		{
			sample_index = j;
		}
		/* instead of saving the whole samples (features of a child), we saved the samples indexes and use them */
		else
		{
			sample_index = group[j];
		}

		if (max_samples[sample_index][feature] < threshold)
		{
			root->Left_group[left_counter] = sample_index;
			left_counter++;
		}
		else
		{
			root->Right_group[right_counter] = sample_index;
			right_counter++;
		}
		root->left_counter = left_counter;
		root->right_counter = right_counter;
	}

	if(group != NULL)
	{
		return root;
	}
}

float gini_index(struct Node*root, int y_train[MEMORY_SIZE+UPDATE_THR])
{
	float first_class_counter = 0;
	float second_class_counter = 0;
	float score, score2;
	float gini = 0;
	int sample_index;

	if (root->left_counter != 0)
	{
		for (int j = 0; j < root->left_counter; j++)
		{
			sample_index = root->Left_group[j];
			if (y_train[sample_index] == classes[0])
			{
				first_class_counter++;
			}
			else
			{
				second_class_counter++;
			}
		}
		score = (first_class_counter/root->left_counter)*(first_class_counter/root->left_counter) + (second_class_counter/root->left_counter)*(second_class_counter/root->left_counter);
		gini = (1.0 - score) * (root->left_counter/(root->left_counter+root->right_counter));
		first_class_counter = 0;
		second_class_counter = 0;
	}
	if (root->right_counter != 0)
	{
		for (int j = 0; j < root->right_counter; j++)
		{
			sample_index = root->Right_group[j];
			if (y_train[sample_index] == classes[0])
			{
				first_class_counter++;
			}
			else
			{
				second_class_counter++;
			}
		}
		score2 = (first_class_counter/root->right_counter)*(first_class_counter/root->right_counter) + (second_class_counter/root->right_counter)*(second_class_counter/root->right_counter);
		gini += (1 - score2) * (root->right_counter/(root->left_counter+root->right_counter));
	}
	return gini;
}

struct Node* get_split(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* root, int* group, int y_train[MEMORY_SIZE+UPDATE_THR], int size)
{
	float b_score = 999;
	float gini, threshold;
	float left_ctr, right_ctr;
	int left_grp[MEMORY_SIZE];
	int right_grp[MEMORY_SIZE];
	int sample_index;

	for(int i = 0; i < N_FEATURE; i++)
	{
		for(int j = 0; j < size; j++)
		{
			if(group == NULL)
			{
				threshold = max_samples[j][i];
			}
			else
			{
				sample_index = group[j];
				threshold = max_samples[sample_index][i];
			}
			split_samples(max_samples, root, group, i, threshold, size);
			gini = gini_index(root, y_train);
			if (gini < b_score)
			{
				root->threshold = threshold;
				root->feature = i;
				left_ctr = root->left_counter;
				right_ctr = root->right_counter;
				memcpy(left_grp, root->Left_group, sizeof(int)*size);
				memcpy(right_grp, root->Right_group, sizeof(int)*size);
				b_score = gini;
			}
		}
	}
	root->left_counter = left_ctr;
	root->right_counter = right_ctr;
	memcpy(root->Left_group, left_grp, sizeof(int)*size);
	memcpy(root->Right_group, right_grp, sizeof(int)*size);
	if(group == NULL)
	{
		split(max_samples, root, y_train, MAX_DEPTH, MIN_SIZE, 0);
	}
	return root;
}

struct Node* GetNewNode()
{
	struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
	newNode->left = newNode->right = NULL;
	return newNode;
}

struct Node* split(float max_samples[MEMORY_SIZE+UPDATE_THR][N_FEATURE], struct Node* node, int y_train[MEMORY_SIZE+UPDATE_THR], int max_depth, int min_size, int depth)
{
	int out;
	if (node->left_counter == 0 || node->right_counter == 0)
	{
		out = to_terminal(node->Left_group, y_train, node->left_counter);
		node->left_class = out;
		out = to_terminal(node->Right_group, y_train, node->right_counter);
		node->right_class = out;
		return node;
	}
	if (depth >= max_depth)
	{
		out = to_terminal(node->Left_group, y_train, node->left_counter);
		node->left_class = out;
		out = to_terminal(node->Right_group, y_train, node->right_counter);
		node->right_class = out;
		return node;
	}
	if(node->left_counter + node->right_counter <= min_size)
	{
		out = to_terminal(node->Left_group, y_train, node->left_counter);
		node->right_class = out;
	}
	else
	{
		node->left = GetNewNode();
		get_split(max_samples, node->left, node->Left_group, y_train, node->left_counter);
		split(max_samples, node->left, y_train, max_depth, min_size, depth+1);
	}
	if(node->right_counter + node->left_counter <= min_size)
	{
		out = to_terminal(node->Right_group, y_train, node->right_counter);
		node->right_class = out;
	}
	else
	{
		node->right = GetNewNode();
		get_split(max_samples, node->right, node->Right_group, y_train, node->right_counter);
		split(max_samples, node->right, y_train, max_depth, min_size, depth+1);
	}
	return node;
}

int to_terminal(int *group, int y_train[MEMORY_SIZE+UPDATE_THR], int size)
{
	int first_class = 0;
	int second_class = 0;
	int sample_index, out;

	for (int i = 0; i < size; i++)
	{
		sample_index = group[i];
		if (y_train[sample_index] == classes[0])
		{
			first_class++;
		}
		else
		{
			second_class++;
		}
	}
	if(first_class > second_class)
	{
		out = classes[0];
	}
	else
	{
		out = classes[1];
	}
	return out;
}
