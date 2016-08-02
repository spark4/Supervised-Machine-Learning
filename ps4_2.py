
import sklearn 
import random
import pylab


# default values to be used unless specified differently
DEFAULT_CORRUPT_FRAC = 0.05
DEFAULT_HOLDOUT_FRAC = 0.2
DEFAULT_THRESHOLD = 0.5
FILENAME = 'tumorInfo.txt'


def corruptData(data, corrupt_frac):
    """
    Returns a copy of the data where roughly corrupt_frac of the
    values have been overwritten with random numbers

    data (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    corrupt_frac (float): a float between 0 and 1, determining the
            probability that a given feature value will be corrupted

    Returns a corrupted version of the data
    """
    newData = []
    for line in data:  
        fields = line.split(',')
        newLine = fields[0] + ',' + fields[1]
        for i in range(2, len(fields)):
            fields[i] = float(fields[i])
            newLine = newLine + ','
            if random.random() < corrupt_frac:
                fields[i] = round(random.gauss(0, 100.0), 5)
            newLine = newLine + str(fields[i])
        newData.append(newLine)
    return newData
       
def printStats(truePos, falsePos, trueNeg, falseNeg, spaces = ''):
    """
    Pretty-print the true/false negatives/positives
    """
    print spaces + 'Accuracy =', accuracy(truePos, falsePos, trueNeg, falseNeg)
    print spaces + 'Sensitivity =', sensitivity(truePos, falseNeg)
    print spaces + 'Specificity =', specificity(trueNeg, falsePos)
    print spaces + 'Pos. Pred. Val. =', posPredVal(truePos, falsePos)

class Tumor(object):
    """
    Wrapper for the tumor data points
    """
    def __init__(self, idNum, malignant, featureNames, featureVals):
        self.idNum = idNum
        self.label = malignant
        self.featureNames = featureNames
        self.featureVals = featureVals
    def distance(self, other):
        dist = 0.0
        for i in range(len(self.featureVals)):
            dist += abs(self.featureVals[i] - other.featureVals[i])**2
        return dist**0.5
    def getLabel(self):
        return self.label
    def getFeatures(self):
        return self.featureVals
    def getFeatureNames(self):
        return self.featureNames
    def __str__(self):
        return str(self.idNum) + ', ' + str(self.label) + ', ' \
               + str(self.featureVals)

def getTumorData(inData, dontUse = []):
    """
    Parses each data point in inData into an instance of the Tumor class,
    where the features listed in dontUse are omitted

    inData (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    dontUse (list): a list of strings, each the name of a feature to omit

    Returns a list of Tumor instances built from the data points provided
            and also returns the list of the names of the features used
    """
    means = ['radiusMean', 'textureMean', 'perimeterMean', 'areaMean',\
             'smoothnessMean', 'compactnessMean', 'concavityMean',\
             'concavePtsMean', 'symmetryMean', 'fractalDMean']
    stdErrs = ['radiusSE', 'textureSE', 'perimeterSE', 'areaSE', \
               'smoothnessSE', 'compactnessSE', 'concavitySE', 
               'concavePtsSE', 'symmetrySE', 'fractalDSE']
    worsts = ['radiusWorst','textureWorst', 'perimeterWorst', 'areaWorst',\
              'smoothnessWorst', 'compactnessWorst', 'concavityWorst',\
              'concavePtsWorst','symmetryWorst', 'fractalDWorst']
    possibleFeatures = means + stdErrs + worsts
    data = []
    for line in inData:
        split = line.split(',')
        idNum = int(split[0])
        if split[1] == 'B':
            malignant = 0
        elif split[1] == 'M':
            malignant = 1
        else:
            raise ValueError('Not B or M')
        featureVec, featuresUsed = [], []
        for i in range(2, len(split)):
            if possibleFeatures[i-2] not in dontUse:
                featureVec.append(float(split[i]))
                featuresUsed.append(possibleFeatures[i-2])
        data.append(Tumor(idNum, malignant, featuresUsed, featureVec))
    return data, featuresUsed


def readData(file_name):
    """
    Reads the data at file file_name into a list of
    strings, each encoding a data point

    file_name (string): name of the data file

    Returns a list of strings, each string encoding
            the ID, label, and features of a tumor
    """
    f = open(file_name)
    lines = f.readlines()
    data = [line.strip() for line in lines]
    f.close()
    return data
    

######################
##      PART 1      ##
######################

def splitData(data, holdout_frac):
    """
    Split data set into training and test sets
    
    data (list): list of Tumor instances
    holdout_frac (float): fraction of data points for testing,
                          as a float in [0,1] inclusive
    
    Returns a tuple (trainingData, testData) where trainingData and
            testData are both lists of Tumor instances and together
            partition the original data list
    
    Randomly split the data into training and test sets, each represented
    as a list, such that the test set takes roughly holdout_frac fraction
    of the original data
    """ 
    
    testset = []
    training= []
    for line in data:  
        if random.random() <= holdout_frac:
            testset.append(line)
        else:
            training.append(line)  
    return tuple([training, testset])


######################
##      PART 2      ##
######################

def trainModel(train_set):
    """
    Trains a logistic regression model with the given dataset

    train_set (list): list of data points of type Tumor

    Returns a model of type sklearn.linear_model.LogisticRegression
            fit to the training data

    """
    data_matrix = []
    i = []
    features = []
    labels_array= [] 
    
    for num in range(len(train_set)):
        i.append(num + 1)
       
    for tumor in train_set:
        feature = tumor.getFeatures()
        features.append(feature)
        label = tumor.getLabel()
        labels_array.append(label)
    data_matrix.append(features)
    data_matrix.append(labels_array)
        
    
    log_reg = sklearn.linear_model.LogisticRegression()
    return log_reg.fit(features, labels_array)    

def predictLabels(model, threshold, data_points):
    """
    Uses the model and probability threshold to predict labels for
    the given data points

    model (LogisticRegression): a trained model
    threshold (float): a value between 0 and 1 to be used as a decision threshold
    data_points (list): list of Tumor objects for which to predict the labels

    Returns a list of labels (value 0 or 1), one for each data point

    """
    labels = []
    features = []
        
    for x in data_points:
        features.append(x.getFeatures())
        
    probs = model.predict_proba(features)
    for tumor in range(len(probs)):
        if probs[tumor][1] > threshold:
            labels.append(1)
        else:
            labels.append(0)

    return labels

    

######################
##      PART 3      ##
######################

def scoreTestSet(model, threshold, test_set):
    """
    Uses the model and threshold to predict labels for the given data points,
    and compares the predicted labels to the true labels of the data.

    model (LogisticRegression): a trained model
    threshold (float): a value between 0 and 1 to be used as a decision threshold
    test_set (list): list of labeled Tumor objects to evaluate the model on

    Returns a tuple with the true positive, false positive, true negative, and
            false negative counts in that order
    """
    trueLabels = []    
   
    predicted = predictLabels(model, threshold, test_set)
   
    for tumor in test_set:
        label = tumor.getLabel()
        trueLabels.append(label)
        
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    
    for label in range(len(predicted)):
        if predicted[label] == 0 and trueLabels[label] == 0:
            trueNeg += 1
        if predicted[label] == 0 and trueLabels[label] == 1:
            falseNeg += 1
        if predicted[label] == 1 and trueLabels[label] == 1:   
            truePos += 1
        if predicted[label] == 1 and trueLabels[label] == 0:
            falsePos += 1
    
    return tuple([truePos, falsePos, trueNeg, falseNeg])
    

def accuracy(truePos, falsePos, trueNeg, falseNeg):
    """
    Fraction of correctly identified elements
    
    truePos (int): number of true positive elements
    falsePos (int): number of false positive elements
    trueNeg (int): number of true negative elements
    falseNeg (int): number of false negative elements
    
    Returns the fraction of true positive or negative elements 
            out of all elements
    """
    total = float(truePos + falsePos + trueNeg + falseNeg)
    return float(truePos + trueNeg) / total    
    
           
def sensitivity(truePos, falseNeg):
    """
    Fraction of correctly identified positive elements out of all positive elements
    
    truePos (int): number of true positive elements
    falseNeg (int): number of false negative elements
    
    Returns the fraction of true positive elements out of all positive elements
    If there are no positive elements, returns a nan
    """
    if (truePos + falseNeg) > 0:
        return float(truePos)/(truePos + falseNeg)
    else:
        return float('nan')


def specificity(trueNeg, falsePos):
    """
    Fraction of correctly identified negative elements out of all negative elements

    trueNeg (int): number of true negative elements
    falsePos (int): number of false positive elements  
    
    Returns the fraction of true negative elements out of all negative elements
    If there are no negative elements, returns a nan
    """
    if (trueNeg + falsePos) > 0:
        return float(trueNeg)/(trueNeg + falsePos)
    else:
        return float('nan')
   
    
def posPredVal(truePos, falsePos):
    """
    fraction of correctly identified positive elements 
    out of all positively identified elements
    
    truePos (int): number of true positive elements
    falsePos (int): number of false positive elements
    
    Returns the fraction of correctly identified positive elements 
            out of all positively identified elements  
    If no elements were identified as positive, returns a nan
    """
    if (truePos + falsePos) > 0:
        return float(truePos)/(truePos + falsePos)
    else:
        return float('nan')
    

def buildROC(model, eval_data):
    """
    Plots the ROC curve, namely the true positive rate (y-axis) vs the false positive rate (x-axis)
    
    model (LogisticRegression): a trained logistic regression model
    eval_data (list): a list of Tumor instances on which to evaluate the model
    
    Returns the area under the curve
    
    Plot the ROC curve as measured in p values in the interval [0,1], inclusive, 
    in 0.01 increments
    At each p value, apply the model and measure the true positive rate 
    and false positive rate.
    
    Remember to give a meaningful title to your plot and include 
    in it the area under the curve 
    """
    truePosRate = []
    falsePosRate = [] 
    xVals = []
    
    num = 0
    while num <= 1:
        xVals.append(num)
        num += 0.05
        
    for p in xVals:
        truePos, falsePos, trueNeg, falseNeg = scoreTestSet(model, p, eval_data)
        true_pos = sensitivity(truePos, falseNeg)
        specificityy = specificity(trueNeg, falsePos)
        false_pos = 1 - specificityy
        truePosRate.append(true_pos)
        falsePosRate.append(false_pos)
    
    pylab.plot(falsePosRate, truePosRate)
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve of Tumor Label Predicting Model')
    pylab.show()
    
    print sklearn.metrics.auc(falsePosRate, truePosRate, reorder = True)
    

def plotPerfVsCorruption(file_name, holdout_frac, threshold):
    """
    Plots model accuracy against the fraction of data that is corrupted

    file_name (string): the name of the file containing the uncorrupted data
    holdout_frac (float): fraction of data points for testing
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    This function does not return anything

    Read the data from the specified file, using readData. For each value of r
    in the interval [0,1], inclusive, in 0.05 increments, generate a version of the data
    with r as the corruption fraction. For each corruption rate, split the corrupted data
    between training and testing sets using the specified holdout fraction, train a model
    on the training set and test on the testing set, using the threshold specified.
    Plot the model accuracy for each value of r
    """
    dataset = readData(file_name)
    xVals = []
    num = 0
    while num <= 1:
        xVals.append(num)
        num += 0.05
        
    accuracy_list = []
    
    for r in xVals:
        corrupt = corruptData(dataset, r)
        new_data, names = getTumorData(corrupt, dontUse = [])
        training, test_set = splitData(new_data, holdout_frac)
        model = trainModel(training)
        truePos, falsePos, trueNeg, falseNeg = scoreTestSet(model, threshold, test_set)
        accuracyVals = accuracy(truePos, falsePos, trueNeg, falseNeg)
        accuracy_list.append(accuracyVals)
    
    pylab.plot(xVals, accuracy_list)
    pylab.xlabel('Corruption Fraction')
    pylab.ylabel('Accuracy')
    pylab.title('Accuracy of model with varying corruption fractions')
    pylab.show()
    

######################
##      PART 4      ##
######################

def findBestFeatureToEliminate(training_data, features_to_omit, threshold):
    """
    Identifies the feature that most improves accuracy when removed. If no
    feature elimination improves over the accuracy using the entire set, then
    the returned feature is None

    training_data (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    features_to_omit (list): a list of strings, each the name of a feature that
            should be omited in the experiment
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    Returns a tuple with the name of the best feature to eliminate and the best
            model. If no feature elimination results in improved accuracy, 
            the value returned as the feature name should be None, 
            and the model returned should be the model trained with 
            all the features

    Note A: the features in features_to_omit should not be used for training or
    evaluating any model. When we say 'model trained with all the features' we
    mean all features EXCEPT those in features_to_omit
    Note B: use the probability threshold specified for predicting labels
    Note C: use a holdout fraction of 0.2 for the development set

    """
    
    
    old_training, old_development = splitData(training_data, 0.2)
    old_model = trainModel(old_training) 
    truePos, falsePos, trueNeg, falseNeg = scoreTestSet(old_model, threshold, old_development)
    old_accuracy = accuracy(truePos, falsePos, trueNeg, falseNeg)
  
#    old_data, featuresUsed = getTumorData(training_data, dontUse = [])
#    old_training, old_development = splitData(old_data, 0.2)
#    old_model = trainModel(old_training) 
#    truePos, falsePos, trueNeg, falseNeg = scoreTestSet(old_model, threshold, old_development)
#    old_accuracy = accuracy(truePos, falsePos, trueNeg, falseNeg)
    
#    lols = []    
#    for tumor in data_set:
#        lols.append(str(tumor))

    
    for feature in features_to_omit:
        new_data, featuresUsed = getTumorData(training_data, dontUse = [feature])
        training, development = splitData(new_data, 0.2)
        model = trainModel(training)
        truePos, falsePos, trueNeg, falseNeg = scoreTestSet(model, threshold, development)
        potential_accuracy = accuracy(truePos, falsePos, trueNeg, falseNeg)
        current_feature = None
        best_model = old_model  
        current_accuracy = old_accuracy
        if current_accuracy < potential_accuracy:
            current_accuracy = potential_accuracy
            current_feature = feature
            best_model = model
    return tuple([current_feature, best_model])
    
    
def buildReducedModel(training_data, threshold):
    """
    Greedily eliminates features until no performance improvement is gained
    from elimination, and returns the best performing model

    training_data (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    Returns a tuple with the model trained on the best performing subset 
            of features along with the final list of features to omit which
            was used when training the model
    """
    features = []

    new_data, features = getTumorData(training_data)
    for tumor in new_data:  
        feature = tumor.getFeatureNames()
        features.append(feature)
        

    copy = features[:]
    
    features_to_omit = []
   
#    print "tHIS IS TRAINING DATA", training_data
    current_feature, best_model = findBestFeatureToEliminate(training_data, features, threshold)
    copy.remove(current_feature)
    potential_feature, potential_model = findBestFeatureToEliminate(training_data, copy, threshold)
    while current_feature != potential_feature:  
        features_to_omit.append(current_feature)
        features.remove(current_feature)
        copy.remove(current_feature)
        current_feature = potential_feature
        best_model = potential_model
        potential_feature, potential_model = findBestFeatureToEliminate(training_data, copy, threshold)
    
    return tuple([best_model, features_to_omit])
    

######################
##      PART 5      ##
######################

def runExperiment(file_name, corrupt_frac, holdout_frac, threshold):
    """
    Trains and evaluates a model using all the features, then trains
    an improved model using feature reduction and evaluates it.

    file_name (string): name of data file
    corrupt_frac (float): fraction of data to be corrupted
    holdout_frac (float): fraction of data to be held out for testing
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    Returns a tuple with the following values, in order:
            the accuracy of the full model evaluated on the training set,
            the accuracy of the full model evaluated on the testing set,
            the accuracy of the reduced model evaluated on the training set,
            and the accuracy of the reduced model evaluated on the testing set
    
    Note: code will also generate plots and evaluation metrics as 
          discussed in the problem description
    """
    dataset = readData(file_name)
    corrupt = corruptData(dataset, corrupt_frac)
    new_data, names = getTumorData(corrupt, dontUse = [])
    training, test_set = splitData(new_data, holdout_frac)
    
    full_training_model = trainModel(training)
    truePos, falsePos, trueNeg, falseNeg = scoreTestSet(full_training_model, threshold, training)
    training_accuracy = accuracy(truePos, falsePos, trueNeg, falseNeg)    
    print "training accuracy:", training_accuracy
    print "training sensitivity:", sensitivity(truePos, falseNeg)
    print "training specificity:", specificity(trueNeg, falsePos)
    print "training pos pred val:", posPredVal(truePos, falsePos)
    
#    full_test_model = trainModel(test_set)
    truePos, falsePos, trueNeg, falseNeg = scoreTestSet(full_training_model, threshold, test_set)
    test_accuracy = accuracy(truePos, falsePos, trueNeg, falseNeg) 
    print "test accuracy:", test_accuracy
    print "test sensitivity:", sensitivity(truePos, falseNeg)
    print "test specificity:", specificity(trueNeg, falsePos)
    print "test pos pred val:", posPredVal(truePos, falsePos)
    
    buildROC(full_training_model, test_set)
    plotPerfVsCorruption(file_name, holdout_frac, threshold)
    
    reduced_training_model, features_to_omit = buildReducedModel(corrupt, threshold)
    truePos, falsePos, trueNeg, falseNeg = scoreTestSet(reduced_training_model, threshold, test_set)
    reduced_training_accuracy_test = accuracy(truePos, falsePos, trueNeg, falseNeg)
    
    truePos, falsePos, trueNeg, falseNeg = scoreTestSet(reduced_training_model, threshold, training)
    reduced_training_accuracy_training = accuracy(truePos, falsePos, trueNeg, falseNeg)
    
    return tuple([training_accuracy, test_accuracy, reduced_training_accuracy_training, reduced_training_accuracy_test])

# Below code will run simulation
if __name__ == '__main__':
    runExperiment(FILENAME, DEFAULT_CORRUPT_FRAC, DEFAULT_HOLDOUT_FRAC, DEFAULT_THRESHOLD)


