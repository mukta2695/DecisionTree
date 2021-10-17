import pandas as pd
import argparse
from pprint import pprint
import numpy as np
eps=np.finfo(float).eps




###defining entropy

def entropy_of_dataset(np_data):    
    class_column = np_data[:, -1]
    class_unique_values, counts= np.unique(class_column, return_counts=True)

    prob=counts/len(class_column)
    entropy = sum(prob*-np.log2(prob))

    #print(entropy)
    #print(prob)
    #print(counts)
    
    return entropy

def find_entropy_attribute(data,attribute):
    Class = data.keys()[-1]   
    target_variables = data[Class].unique()  
    variables = data[attribute].unique()    
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(data[attribute][data[attribute]==variable][data[Class] ==target_variable])
            den = len(data[attribute][data[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*np.log2(fraction+eps)
        fraction2 = den/len(data)
        entropy2 += -fraction2*entropy
    return abs(entropy2)
  

def wt_var(data,split_attr_name):
    #print(split_attr_name)
    Class=data['Class']
    ds=data[split_attr_name]
    #print(len(Class))
    #print(ds)
    k=len(ds)
    k0=0
    k1=0
    weighted_var=0
    indices=ds.index
    for var in indices:
        #print(ds[index])
        if ds[var]==0:
            #print(index)
            if Class[var]==0:
                k0+=1
        elif ds[var]==1:
            if Class[var]==1:
                k1+=1
    prob1=k0/(k+eps)
    prob2=k1/(k+eps)
    weighted_var=(prob1*((k0*k1)/((k*k))+eps))+(prob2*((k0*k1)/((k*k))+eps))
    return weighted_var

def var_of_ds(data):
    Class=data.keys()[-1]
    k0=0
    k1=0
    for var1 in data[Class]:
        if var1==0:
            k0+=1
        elif var1==1:
            k1+=1
    k=k0+k1
    var_ds=(k0*k1)/((k*k)+eps)
    return var_ds
    
def var_gain(ds,split_attribute_name):
    #print(ds)
    total_variance=var_of_ds(ds)
    weighted_var=wt_var(ds, split_attribute_name)
    varGain= total_variance-weighted_var
    return varGain

    
def information_gain(ds,split_attribute_name):
    total_entropy = entropy_of_dataset(ds.values)
    vals,counts = np.unique(ds[split_attribute_name],return_counts=True)
    weighted_Entropy = find_entropy_attribute(ds, split_attribute_name)   
    info_gain = total_entropy-weighted_Entropy
    return info_gain

def decision_tree(data,features,heuristic,att_name="Class", node_class=None):
    heu=heuristic
    #Check the purity, that is check if all target values have the same value. If yes then classify it as that value
    if len(np.unique(data[att_name])) == 1:
        return np.unique(data[att_name])[0]

    else:
        node_class = np.unique(data[att_name])[np.argmax(np.unique(data[att_name],return_counts=True)[1])]

    #Select the feature which best splits the dataset
    if heuristic=='Entropy':
        info_gain_values = [information_gain(data,feature)for feature in features] #Return the infgain values
        #print(info_gain_values)
        if len(info_gain_values)==0:
            return({})
        else:
            best_index = np.argmax(info_gain_values)
            best_feature = features[best_index]
    elif heuristic=='Variance':
        variance_gain = [var_gain(data, feature) for feature in features]
        #print(variance_gain)
        if len(variance_gain)==0:
            return({})
        else:
            index_of_max = variance_gain.index(max(variance_gain)) 
            best_feature = features[index_of_max]
        #print(best_feature)

    #Create the tree structure
    tree = {best_feature:{}}

    #Remove the feature with the best info gain
    features = [i for i in features if i!= best_feature]

    #Grow the tree branch under the root node

    for value in np.unique(data[best_feature]):
        sub_data = data.where(data[best_feature]==value).dropna()
        #print(sub_data)
        subtree = decision_tree(sub_data,features,heu,att_name,node_class)
        #Add the subtree
        tree[best_feature][value] = subtree
    return(tree)

#Predict
def pred(query,tree,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result,dict):
                return pred(query,result)
            else:
                return result
##check the accuracy

def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])

   #calculation of accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = pred(queries[i],tree,1.0)
    print("The Prediction accuracy is:",(np.sum(predicted["predicted"]==data["Class"])/len(data)))
  
 
def printTree(tree, d=0):
    if (tree == None or len(tree) == 0):
        print("\t" * d, "-")
    else:
        for key, val in tree.items():
            if (isinstance(val, dict)):
                print ("\n|" * d, key)
                printTree(val, d+1)
            else:
                print('\t' * d,'{} : {}'.format(key,val))
                
#tree=decision_tree(training_set,training_set.columns[:-1],'Entropy')
#pprint(tree)
#printTree(tree)
#test(test_set,tree)