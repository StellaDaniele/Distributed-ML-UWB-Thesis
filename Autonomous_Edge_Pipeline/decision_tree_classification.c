/* ----------------------------------------------------------------------
 * Project:      Autonomous Edge Pipeline
 * Title:        decision_tree_classification.c
 * Description:  Decision Tree classifier
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include "main.h"
#include "dataset.h"
#include "decision_tree_training.h"

/**
 * @brief decision_tree_classification
 * @param[in]       root                decision tree root node
 * @param[in]       X                   pointer to input sample
 * @return     The function returns the best class for the X sample
 */

int decision_tree_classifier(struct Node* root, float X[])
{
    if(X[root->feature] < root->threshold)
	{
		if(root->left != NULL)
		{
			decision_tree_classifier(root->left, X);
		}
		else
		{
			return root->left_class;
		}
		
	}
	else
	{
		if(root->right != NULL)
		{
			decision_tree_classifier(root->right, X);
		}
		else
		{
			return root->right_class;
		}
	}
}