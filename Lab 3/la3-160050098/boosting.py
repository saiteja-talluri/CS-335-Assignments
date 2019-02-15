import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        
        weights = np.full(len(trainingData),1.0, dtype=float)
        weights = weights/np.sum(weights)

        for i in range(len(self.classifiers)):
            error = 0
            self.classifiers[i].train(trainingData,trainingLabels, weights)
            guesses = self.classifiers[i].classify(trainingData)

            for j in range(len(guesses)):
                if guesses[j] != trainingLabels[j]:
                    error += weights[j]

            for j in range(len(guesses)):
                if guesses[j] == trainingLabels[j]:
                    weights[j] *= error/(1-error)

            weights = weights/np.sum(weights)
            self.alphas[i] = float(np.log((1-error)/error))

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        guesses = np.zeros(len(data),dtype=float)
        
        
        for i in range(len(self.classifiers)):
            guesses += np.multiply(self.classifiers[i].classify(data),self.alphas[i])

        guesses = np.sign(guesses)
        guesses = guesses.astype(np.int)

        for i in range(len(guesses)):
            if guesses[i] == 0:
                guesses[i] = np.random.choice(self.classifiers[0].legalLabels)

        return guesses