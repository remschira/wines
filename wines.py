'''
cite:

1. Building A Logistic Regression in Python, Step by Step,
https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

2. Principal Component Analysis
http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html

3. SCIKIT-LEARN : DATA COMPRESSION VIA DIMENSIONALITY REDUCTION I - PRINCIPAL COMPONENT ANALYSIS (PCA)
http://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Compresssion_via_Dimensionality_Reduction_1_Principal_component_analysis%20_PCA.php

'''

import numpy as np
import pandas as pd
from wines_functions import parseConfigFile,plotData,\
    featureMeanAtScore,trainTestSplit,scaleData,\
    explainedVariancePlot,principalComponents,PCAplot,testClassifiers,\
    importanceFeaturesPlot,outlierPCAplot,outlierDetector,\
    scaleDataPlots,importanceFeatures,trimData



if __name__ == '__main__':

    rng        = np.random.RandomState(42) #for reproducibility

    ''' 
    Read the wines.ini file to set the output path. And to
    set the classifier.
    '''
    path       = './'
    filename   = 'wines.ini'
    IC         = parseConfigFile(path,filename)

    outputPath = IC['output']


    ''' 
    import the red wine data
    '''
    website = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data    = pd.read_csv(website,delimiter=';')


    '''
    create an array of all the wine features
    '''
    y        = ['quality']
    features = [i for i in data if i not in y] #this used in importanceFeaturesPlot


    
    plotData(outputPath,data,'pH')#saves plots in output dir of a histo of
                                  #pH and the feature correlation heat map,and other plots


    
    featureMeanAtScore(outputPath,data)#prints to file the means of each feature at each wine score

    '''
    split the data into training and test sets. And scale the data.
    '''
    X_train,X_test,y_train,y_test = trainTestSplit(data,rng)
    X_train,X_test                = scaleData(X_train,X_test)

    scaleDataPlots(outputPath,features,X_train)


    '''
    label outliers and remove from training set if the ini file says so.
    '''    
    detector_name  = IC['detector']
    outliers       = outlierDetector(detector_name,rng,X_train)    
    removeOutliers = IC['remove']    
    if removeOutliers == 'True':
        X_train    = X_train[outliers==1]
        y_train    = y_train[outliers==1]



    '''
    creates PCA plots. saved to output dir
    '''
    explainedVariancePlot(outputPath,X_train)
    components             = int(IC['pca_components'])
    X_train_pca,X_test_pca = principalComponents(components,X_train,X_test)
    if components == 2 or components == 3:
        PCAplot(outputPath,components,X_train_pca,y_train)
        if removeOutliers == 'False':#this only works when False b/c
                                    #len(outliers) > len(X_train_pca) otherwise
            outlierPCAplot(outputPath,components,X_train_pca,outliers)                


            
    '''
    Make importance features plot for random forest
    '''
    importanceFeaturesPlot(outputPath,features,X_train,X_test,y_train,y_test)



    imp = False #are we going to use feature importances to remove unimportant features
    if imp == True:
        '''
        Remove from the full dataset the features that have importance less than 8%. Now we
        have fewer features. Split this new data into training and test. And run the classifiers
        '''
        importances                   = importanceFeatures(features,X_train,X_test,y_train,y_test)    
        data_trimmed                  = trimData(data,importances)    
        X_train,X_test,y_train,y_test = trainTestSplit(data_trimmed,rng)
        X_train,X_test                = scaleData(X_train,X_test)
        
        testClassifiers(outputPath,imp,removeOutliers,X_train,X_test,y_train,y_test)

    else:
        testClassifiers(outputPath,imp,removeOutliers,X_train,X_test,y_train,y_test)

        



    


    


