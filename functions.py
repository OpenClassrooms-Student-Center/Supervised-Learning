import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import graphviz 
from sklearn import tree

def classComparePlot(self, class_name, plotType='density'):
    '''Show comparative plots comparing the distribution of each feature for each class.  plotType can be 'density' or 'hist' '''
    numcols = len(self.columns) - 1

    unit_size = 5
    classes = self[class_name].nunique()           # no of uniques classes
    class_values = self[class_name].unique()       # unique class values

    print('Comparative histograms for',class_values)

    colors = plt.cm.get_cmap('tab10').colors
    fig = plt.figure(figsize=(unit_size,numcols*unit_size))
    ax = [None]*numcols 
    i = 0
    for col_name in self.columns:
        minVal = self[col_name].min()
        maxVal = self[col_name].max()
        
        if col_name != class_name:                
            ax[i] = fig.add_subplot(numcols,1,i+1)   
            for j in range(classes):   
                selectedCols = self[[col_name,class_name]]
                filteredRows = selectedCols.loc[(self[class_name]==class_values[j])]
                values = filteredRows[col_name]
                values.plot(kind=plotType,ax=ax[i],color=[colors[j]], alpha = 0.8, label=class_values[j], range=(minVal,maxVal))
                ax[i].set_title(col_name)
                ax[i].grid()                                  
                #(self[[col_name,class_name]].loc[(self[class_name]==class_values[j])])[[col_name]].hist(ax=ax[i],color=[colors[j]], alpha = 0.5, label=class_values[j])
            ax[i].legend()
            i += 1        

    plt.show()

def boxPlotAll(df):
        '''Show box plots for each feature'''
        df = df.select_dtypes(include=[np.number])
        data_cols = len(df.columns)
        unit_size = 5
        layout_cols = 4
        layout_rows = int(data_cols/layout_cols+layout_cols)
        df.plot(kind='box', subplots=True, figsize=(layout_cols*unit_size,layout_rows*unit_size), layout=(layout_rows,layout_cols))

        plt.show()   
        
def histPlotAll(self):
        '''Show histograms for each feature'''
        df = df.select_dtypes(include=[np.number])
        data_cols = len(self.columns)
        unit_size = 5
        layout_cols = 4
        layout_rows = int(data_cols/layout_cols+layout_cols)
        self.hist(figsize=(layout_cols*unit_size,layout_rows*unit_size), layout=(layout_rows,layout_cols))
                
        plt.show()               

def correlationMatrix(df):
    '''Show a correlation matrix for all features.'''
    columns = df.select_dtypes(include=['float64','int64']).columns
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none',cmap='RdYlBu')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation = 90)
    ax.set_yticklabels(columns)
    plt.show()            

def scatterMatrix(df):
    '''Show a scatter matrix of all features.'''
    unit_size = 5
    pd.plotting.scatter_matrix(df,figsize=(unit_size*4, unit_size*4))
    plt.show()
        

def appendEqualCountsClass(df, class_name, feature, num_bins, labels):
    '''Append a new class feature named 'class_name' based on a split of 'feature' into clases with equal sample points.  Class names are in 'labels'.'''
    percentiles = np.linspace(0,100,num_bins+1)
    bins = np.percentile(df[feature],percentiles)
    n = pd.cut(df[feature], bins = bins, labels=labels, include_lowest=True)
    c = df.copy()
    c[class_name] = n
    return c    

def logisticRegressionSummary(model, column_names):
    '''Show a summary of the trained logistic regression model'''
    numclasses = len(model.classes_)
    if len(model.classes_)==2:
        classes =  [model.classes_[1]] # if we have 2 classes, sklearn only shows one set of coefficients
    else:
        classes = model.classes_
    for i,c in enumerate(classes):
        fig = plt.figure(figsize=(8,len(column_names)/3))
        fig.suptitle('Logistic Regression Coefficients for Class ' + str(c), fontsize=16)
        rects = plt.barh(column_names, model.coef_[i],color="lightblue")
        
        for rect in rects:
            width = round(rect.get_width(),4)
            plt.gca().annotate('  {}  '.format(width),
                        xy=(0, rect.get_y()),
                        xytext=(0,2),  
                        textcoords="offset points",  
                        ha='left' if width<0 else 'right', va='bottom')        
        plt.show()
        #for pair in zip(X.columns, model_lr.coef_[i]):
        #    print (pair)

def decisionTreeSummary(model, column_names):
    '''Show a summary of the trained decision tree model'''
    fig = plt.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Decision tree feature importance', fontsize=16)
    rects = plt.barh(column_names, model.feature_importances_,color="khaki")

    for rect in rects:
        width = round(rect.get_width(),4)
        plt.gca().annotate('  {}  '.format(width),
                    xy=(width, rect.get_y()),
                    xytext=(0,2),  
                    textcoords="offset points",  
                    ha='left', va='bottom')    

    plt.show()

def linearRegressionSummary(model, column_names):
    '''Show a summary of the trained linear regression model'''
    fig = plt.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Linear Regression Coefficients', fontsize=16)
    rects = plt.barh(column_names, model.coef_,color="lightblue")

    for rect in rects:
        width = round(rect.get_width(),4)
        plt.gca().annotate('  {}  '.format(width),
                    xy=(0, rect.get_y()),
                    xytext=(0,2),  
                    textcoords="offset points",  
                    ha='left' if width<0 else 'right', va='bottom')        
    plt.show()


def viewDecisionTree(model, column_names):
        dot_data = tree.export_graphviz(model, out_file=None,
                feature_names=column_names,
                class_names=model.classes_,
                filled=True, rounded=True,
                special_characters=True)
        graph = graphviz.Source(dot_data) 
        return graph    


def find_outliers(feature):
    quartile_1, quartile_3 = np.percentile(feature, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((feature > upper_bound) | (feature < lower_bound))