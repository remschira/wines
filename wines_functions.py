import numpy as np
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy.stats import normaltest,ttest_ind,ks_2samp

def parseConfigFile(path, fileName):
    #citation: https://wiki.python.org/moin/ConfigParserExamples
    
    Config       = configparser.ConfigParser()    
    Config.read(path + fileName)

    sections     = Config.sections()

    dict_IC      = {}
    for section in sections:

        options  = Config.options(section)
        for option in options:
            try:
                dict_IC[option] = Config.get(section, option)
                if dict_IC[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict_IC[option] = None


    return dict_IC #this is a dictionary. access parameter value by .e.g. dict_IC['output']


def plotData(path,data,name):
#heatmap idea from:
#https://towardsdatascience.com/predict-employee-turnover-with-python-da4975588aa3

    target   = data['quality'] #the score of the wine. 
    sns.countplot(x=target,palette='hls')
    plt.savefig(path + 'quality_noOutliers_histo.png',bbox_inches='tight',dpi=None)

    
    #This block graphs a histogram of the pH values
    g       = sns.factorplot(x=name, data=data, kind="count")
    g.set_xticklabels(rotation=90)
    plt.savefig(path + name + '_histo.png',bbox_inches='tight',dpi=None)
    plt.clf() #clears plot window, so that new plots have clean canvas

    
    #This gives a heat map of the features. The rotations make the labels readable
    g = sns.heatmap(data.corr())
    for item in g.get_xticklabels():
        item.set_rotation(90)
    for item in g.get_yticklabels():
        item.set_rotation(0)
    plt.savefig(path + 'feature_heatMap.png',bbox_inches='tight',dpi=None)        
    plt.clf()
    
    data.hist(bins=30)
    plt.tight_layout()#fit subplot in figure area
    plt.savefig(path + 'allDataHisto.png',bbox_inches='tight',dpi=None)            
    plt.clf()

    data.boxplot(rot=90,sym='')#sym='' removes the outliers from the plot
    plt.ylim([0,125])
    plt.tight_layout()#fit subplot in figure area
    plt.savefig(path + 'allDataBoxPlot.png',bbox_inches='tight',dpi=None)            
    plt.clf()    

    
def featureMeanAtScore(path,data):
#citation: https://stackoverflow.com/a/7152903

    #redirects print to file
    orig_stdout = sys.stdout
    f           = open(path+'featureMeans.dat','w')
    sys.stdout  = f
    print(data.groupby('quality').mean())#This returns the means of each feature at each wine score
    sys.stdout  = orig_stdout
    f.close()


def trainTestSplit(data,rng):

    y = ['quality']
    X = [i for i in data if i not in y] #removes selected columns

    #print(data[X].head())
    #print(data[y].head())

    X_train,X_test,y_train,y_test = train_test_split(
        data[X],data[y],test_size=0.33,random_state=rng)
    
    return X_train,X_test,y_train,y_test

def scaleData(X_train,X_test):
    
    scaler       = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train      = scaler.transform(X_train) #this converts pandas dataframe to numpy
    #validation_data = scaler.transform(validation_data)
    X_test       = scaler.transform(X_test)
    
    return X_train,X_test

def scaleDataPlots(path,features,X_train):

    #make scaled X_train into pandas df and plot boxplot and histos
    df_X_train = pd.DataFrame()

    for i in range(len(features)):
        df_X_train[features[i]] = X_train[:,i]                
    df_X_train.boxplot(rot=90,sym='')
    plt.ylim([-3,3])    
    #plt.tight_layout()#fit subplot in figure area
    #plt.show()    
    plt.savefig(path + 'scaledTrainDataBoxPlot.png',bbox_inches='tight',dpi=None)            
    plt.clf()
    


def explainedVariancePlot(path,X_train):
#citations:
#1. Principal Component Analysis
#http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
#2. SCIKIT-LEARN : DATA COMPRESSION VIA DIMENSIONALITY REDUCTION I - PRINCIPAL COMPONENT ANALYSIS (PCA)
#http://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Compresssion_via_Dimensionality_Reduction_1_Principal_component_analysis%20_PCA.php


    cor_mat            = np.cov(X_train.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat)#eigen-decomposition of covariance matrix
    eig_pairs          = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))] #list of eigenvalue,
                                                                                              #eigenvector tuples
    eig_pairs.sort(key=lambda x: x[0], reverse=True) #sort the (eigenvalue,eigenvector)
                                                     #tuples list by eigenvalues


    tot         = sum(eig_vals)
    var_exp     = [(i / tot) for i in sorted(eig_vals, reverse=True)] # explained variance of ith eigenvalue =
                                                                      # \lambda_i/sum(lambda)
    cum_var_exp = np.cumsum(var_exp)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1,12), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1,12), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()#fit subplot in figure area
    plt.savefig(path + 'explainedVariance.png',bbox_inches='tight',dpi=None)        
    plt.clf()

def principalComponents(number,X_train,X_test):
#citations:
#1. Principal Component Analysis
#http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
#2. SCIKIT-LEARN : DATA COMPRESSION VIA DIMENSIONALITY REDUCTION I - PRINCIPAL COMPONENT ANALYSIS (PCA)
#http://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Compresssion_via_Dimensionality_Reduction_1_Principal_component_analysis%20_PCA.php

    pca         = PCA(n_components=number)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.fit_transform(X_test)    

    return X_train_pca,X_test_pca
    
def PCAplot(path,components,X_pca,y_train):
#citations:
#1. Principal Component Analysis
#http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
#2. SCIKIT-LEARN : DATA COMPRESSION VIA DIMENSIONALITY REDUCTION I - PRINCIPAL COMPONENT ANALYSIS (PCA)
#http://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Compresssion_via_Dimensionality_Reduction_1_Principal_component_analysis%20_PCA.php
#3. In which I implement Anomaly Detection for a sample data set from Andrew Ng's Machine Learning Course.
#http://sdsawtelle.github.io/blog/output/week9-anomaly-andrew-ng-machine-learning-with-python.html
    
    if components == 2:
        #This block creates the PCA graph    
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111)
        for lab, color in zip((1,2,3,4,5,6,7,8,9,10),
                              ('pink','gray','blue', 'red', 'green','black','brown','yellow','beige','indigo')): 
            ax.scatter(X_pca[y_train.values.ravel()==lab, 0],
                        X_pca[y_train.values.ravel()==lab, 1],
                        label=lab,
                        c=color)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend(loc='best')
        fig.tight_layout()
        plt.savefig(path + 'PCA_2D.png',bbox_inches='tight',dpi=None)        
        plt.clf()
        
    if components == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for lab, color in zip((1,2,3,4,5,6,7,8,9,10),
                              ('pink','gray','blue', 'red', 'green','black','brown','yellow','beige','indigo')): 
            ax.scatter(X_pca[y_train.values.ravel()==lab, 0],
                       X_pca[y_train.values.ravel()==lab, 1],
                       X_pca[y_train.values.ravel()==lab, 2],
                       label=lab,
                       c=color)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(loc='best')    
        fig.tight_layout()
        #plt.show()
        plt.savefig(path + 'PCA_3D.png',bbox_inches='tight',dpi=None)        
        plt.clf()


def outlierPCAplot(path,components,X_pca,outliers):
#citations:
#1. In which I implement Anomaly Detection for a sample data set from Andrew Ng's Machine Learning Course.
#http://sdsawtelle.github.io/blog/output/week9-anomaly-andrew-ng-machine-learning-with-python.html
        

    if components == 2:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.scatter(X_pca[outliers==1, 0],
                   X_pca[outliers==1, 1],
                   label='inliers',c='blue',lw=2,s=2)
        
        ax.scatter(X_pca[outliers==-1, 0],
                   X_pca[outliers==-1, 1],
                   label='outliers',c='red',marker='x',lw=2,s=60)
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend(loc='best')    
        fig.tight_layout()
        #plt.show()
        plt.savefig(path + 'outlierPCA_2D.png',bbox_inches='tight',dpi=None)        
        plt.clf()
    
    if components == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_pca[outliers==1, 0],
                   X_pca[outliers==1, 1],
                   X_pca[outliers==1, 2],
                   label='inliers',c='blue',lw=2,s=2)
        
        ax.scatter(X_pca[outliers==-1, 0],
                   X_pca[outliers==-1, 1],
                   X_pca[outliers==-1, 2],
                   label='outliers',c='red',marker='x',lw=2,s=60)
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(loc='best')    
        fig.tight_layout()
        #plt.show()
        plt.savefig(path + 'outlierPCA_3D_iforest.png',bbox_inches='tight',dpi=None)        
        plt.clf()

def trimData(data,importances):
#exclude all features for which the importance from the rf classifier is less than 0.08
#    r = []
#    for i in range(len(importances)):
#        if importances[i] < 0.08:
#            r.append(i)
    
    exclude      = ['fixed acidity', 'citric acid', 'residual sugar',
                    'free sulfur dioxide', 'pH']
    data_trimmed = data.drop(exclude,1)#1 = column, 0 = row

    return data_trimmed
    


def importanceFeatures(features,X_train,X_test,y_train,y_test):


    clf_rf      = RandomForestClassifier(n_estimators=300,max_depth = 150)    
    clf_rf.fit(X_train,y_train.values.ravel())
    importances = clf_rf.feature_importances_

    return importances


        
def importanceFeaturesPlot(path,features,X_train,X_test,y_train,y_test):


    figurename    = path + 'rf_importanceFeatures.png'


    clf_rf        = RandomForestClassifier(n_estimators=300)    
    clf_rf.fit(X_train,y_train.values.ravel())
    importances   = clf_rf.feature_importances_
    g             = sns.barplot(x=features, y=importances)

    for item in g.get_xticklabels():
        item.set_rotation(90)
    plt.savefig(figurename,bbox_inches='tight',dpi=None)
    plt.clf()
    
    
def testClassifiers(path,imp,removeOutliers,X_train,X_test,y_train,y_test):

    if imp == True:
        file = open(path + 'classifier_performance_trimmed.dat','w')
    elif removeOutliers == 'True':
        file = open(path + 'classifier_performance_noOutliers.dat','w')
    else:
        file = open(path + 'classifier_performance.dat','w')
        



    #The next two lines are for cross validation
    c,r = y_train.shape 
    y_train_reshape = y_train.values.reshape(c,) #I need shape (1071,) not (1071,1) for cross val    

    file.write( '::::::logistic regression::::::\n')    
    lam = [0.001,0.01,0.1,1.,10.,100.,1000.]
    for l in lam:
        clf_logit = LogisticRegression(multi_class='multinomial',solver='newton-cg',C=l)
        clf_logit.fit(X_train,y_train.values.ravel())
        pred_logit = clf_logit.predict(X_test)
        file.write( '\n \n::: with C = %f :::::: \n\n' %l )            
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_logit))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_logit) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_logit,X_train,y_train_reshape,cv=5, scoring='accuracy')))


    file.write( '\n::::::Decision Tree::::::\n')            
    lam = [2,5,8,10,15,30,50,70,100,150]
    for l in lam:
        clf_tree = DecisionTreeClassifier(max_depth=l)
        clf_tree.fit(X_train,y_train.values.ravel())
        pred_tree = clf_tree.predict(X_test)
        file.write( '\n \n::: with max_depth = %d :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_tree))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_tree) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_tree,X_train,y_train_reshape,cv=5, scoring='accuracy')))

    file.write( '\n::::::random forest::::::\n')
    lam = [2,5,8,10,15,30,50,70,100,150]
    #n_estimators = 300 best, acc=.669,trace=353
    #default max_depth is best
    for l in lam:
        clf_rf = RandomForestClassifier(n_estimators=300,max_depth=l)    
        clf_rf.fit(X_train,y_train.values.ravel())
        pred_rf = clf_rf.predict(X_test)
        file.write( '\n \n::: with max_depth = %d :::::: \n\n' %l )        
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_rf))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_rf) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_rf,X_train,y_train_reshape,cv=5, scoring='accuracy')))        

    file.write( '\n::::::Adaboost::::::\n')
    lam = [1,2,4,10,50,100]    
    for l in lam:
        clf_ada = AdaBoostClassifier(n_estimators=l)
        clf_ada.fit(X_train,y_train.values.ravel())
        pred_ada = clf_ada.predict(X_test)

        file.write( '\n \n::: with n_estimators = %d :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_ada))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_ada) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_ada,X_train,y_train_reshape,cv=5, scoring='accuracy'))) 

    
    file.write( '\n::::::gradient boosting::::::\n')
    lam = [1,2,3,5,6,10,15,20,30,50,100]
    #random forest
    #n_estimators = 300 best, acc=.669,trace=353
    #default max_depth is best

    #gradient boosting
    #n_estimators = 300 best, acc=.652,trace=344
    #default max_depth is best
    for l in lam:
        clf_gbm = GradientBoostingClassifier(n_estimators=300,max_depth=l)    
        clf_gbm.fit(X_train,y_train.values.ravel())
        pred_gbm = clf_gbm.predict(X_test)
        file.write( '\n \n::: with max_depth = %d :::::: \n\n' %l )        
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_gbm))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_gbm) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_gbm,X_train,y_train_reshape,cv=5, scoring='accuracy'))) 

    file.write( '\n::::::Linear SVM::::::\n')
    lam = [0.001,0.01,0.02,0.04,0.06,0.08,0.10,1.0,5.0]    
    for l in lam:
        clf_lsvm = LinearSVC(C=l)
        clf_lsvm.fit(X_train,y_train.values.ravel())
        pred_lsvm = clf_lsvm.predict(X_test)
        file.write( '\n \n::: with C = %f :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_lsvm))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_lsvm) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_lsvm,X_train,y_train_reshape,cv=5, scoring='accuracy'))) 

    file.write( '\n::::::SVM rbf::::::\n')
    lam = [0.001,0.01,0.02,0.04,0.06,0.08,0.10,1.0,5.0]    
    for l in lam:
        clf_kernel_svm = SVC(kernel='rbf',C=l)
        clf_kernel_svm.fit(X_train,y_train.values.ravel())
        pred_kernel_svm = clf_kernel_svm.predict(X_test)    
        file.write( '\n \n::: with C = %f :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_kernel_svm))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_kernel_svm) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_kernel_svm,X_train,y_train_reshape,cv=5, scoring='accuracy')))


    file.close()    


def outlierDetector(clf_name,rng,X_train):

    outliers_fraction = 0.04
    if clf_name == 'RobustCovariance':
        clf_ell       = EllipticEnvelope(contamination = outliers_fraction)
        clf_ell.fit(X_train)
        anomaly_score = clf_ell.decision_function(X_train)
        outliers      = clf_ell.predict(X_train)

    if clf_name == 'IsolationForest':
        clf_iforest   = IsolationForest(n_estimators=100,random_state=rng,contamination = outliers_fraction)
        clf_iforest.fit(X_train)
        anomaly_score = clf_iforest.decision_function(X_train)
        outliers      = clf_iforest.predict(X_train)
        

    return outliers
        
