#ifndef DECISION_TREE_CLASSIFIER_H

#define DECISION_TREE_CLASSIFIER_H

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include "main.h"
#include "test.h"
#include "decision_tree_training.h"

int decision_tree_classifier(struct Node* root, float X[]);

#endif