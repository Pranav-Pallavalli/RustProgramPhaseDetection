from analyzer_class import analyzer_class
import os
from GMM import GMM
import numpy as np


def main(): 
    # obj = analyzer_class("/Users/pranavpallavalli/Desktop/Thesis_analyzer/traces/trace.txt")
    # print("The trace list is: ")
    # print("\n")
    # print(obj.preprocess())
    # print("\n")
    # print("The Feature vector generated is: ")
    # print("\n")
    # print(obj.create_feature_vec())
    #print(obj.create_feature_vec())
    # feature_space,all_traces = create_training_data()
    # obj = analyzer_class(None,feature_space,all_traces)
    # print(obj.create_feature_vector_2())
    # feature_space,all_traces = GMM().create_training_data()
    # obj = analyzer_class(None,feature_space,all_traces)
    # print("Feature vector generated is: ")
    # print(obj.create_feature_vec_with_time())
    gmm = GMM()
    # global_feature_spaces, all_traces_list = gmm.create_training_data()
    # print("Global features list: ")
    # print()
    # print("ALL trace files are: ")
    # print(len(all_traces_list[0]))
    # print(len(all_traces_list[0]))
    result = gmm.classify()
    # print("filtered feature vectors are \n")
    # print(new_filtered_features)
    # print("Old feature vectors are \n")
    # print(feature_vectors)
    
    print("Final result is - \n")
    print(result)
    
    # new_feature_vectors = gmm.correlation(phase_changes,feature_vectors)
    # inter_feature_selection_vector = gmm.inter_feature_selection(new_feature_vectors)
    # print(phase_changes)
    # print(gmm.train_gmm_with_kfold_cv(inter_feature_selection_vector))
  
  
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 