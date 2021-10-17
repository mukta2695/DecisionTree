import pandas as pd
import argparse
#import numpy as np
from pprint import pprint
#eps=np.finfo(float).eps

import decisionTree as dt



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('training-set')
    parser.add_argument('validation-set')
    parser.add_argument('test-set')
    parser.add_argument('to-print', choices=['yes', 'no'])
    parser.add_argument('heuristic', choices=['Entropy', 'Variance'])
    args = vars(parser.parse_args())

    training_set = pd.read_csv(r'training_set.csv')
    validation_set = pd.read_csv(r'validation_set.csv')
    test_set = pd.read_csv(r'test_set.csv')

    #label = 'Class'

    
    print('\n--------------------------')
    heuristic=args['heuristic']
    tree = dt.decision_tree(training_set, training_set.columns[:-1], heuristic)
    if args['to-print'] == 'yes':
        print("Tree in dictionary format\n")
        pprint(tree)
        print("Tree in given format\n")
        dt.printTree(tree)
    accuracy_percentage= dt.test(test_set, tree)
    #break

    print('\n--------------------------')

if __name__ == '__main__':
    main()