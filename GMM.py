from math import inf
import numpy as np
from sklearn.mixture import GaussianMixture
from analyzer_class import analyzer_class
from scipy.stats import pearsonr
import os
from sklearn.model_selection import KFold
from math import inf
from sklearn.model_selection import GroupKFold


class GMM:
    # This method processes trace files from the specified directory to create a training dataset. 
    # It creates a unique mapping for each event type (feature_space) and 
    # aggregates all preprocessed traces (all_traces). 
    # This step is crucial for converting raw trace data into a structured format suitable for analysis
    # Also creates and uses global feature space to avoid issues in dimensionality
    # def create_training_data(self):
    #     directory_path = '/Users/pranavpallavalli/Desktop/Thesis_analyzer/traces'
    #     analyzer_objs = []
    #     all_traces = []
    #     feature_space = dict()  # Dictionary to map unique elements to indices also used to create the global feature space
    #     index = 0
    #     # Iterating over each file in the directory
    #     for filename in os.listdir(directory_path):
    #         file_path = os.path.join(directory_path, filename)
    #         print(file_path)
    #          # Creating an analyzer object for each file and appending to the list
    #         analyzer_objs.append(analyzer_class(file_path))
    #     # Preprocessing each file and appending the processed trace to all_traces
    #     for analyzer_obj in analyzer_objs:
    #         preprocessed_file = analyzer_obj.preprocess()
    #         all_traces.append(preprocessed_file)
    #     #Creates global feature space
    #     for trace in all_traces:
    #         for element in trace:
    #             if element not in feature_space:
    #                 feature_space[element] = index
    #                 index = index + 1
    #     print(feature_space)
    #     #returns the processed trace files and the global feature space for training
    #     return feature_space,all_traces

    def squash(self,feature_vector):
        ret = []
        for feature_arr in feature_vector:
            for feature in feature_arr:
                ret.append(feature)
        return ret
    
    def create_training_data(self):
        directory_path = "/Users/pranavpallavalli/Desktop/Thesis_analyzer/traces"
        subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        print(subdirectories)
        global_feature_spaces = []
        all_traces_list = []
        filenames =[]

        for subdir in subdirectories:
            subdir_path = os.path.join(directory_path, subdir)
            analyzer_objs = []
            all_traces = []
            feature_space = dict()
            index = 0

            # Iterating over each .trace file in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith('.trace') or filename.endswith('.txt'):
                    file_path = os.path.join(subdir_path, filename)
                    filenames.append(filename)
                    # Creating an analyzer object for each file and appending to the list
                    analyzer_objs.append(analyzer_class(file_path))

            # Preprocessing each file and appending the processed trace to all_traces
            for analyzer_obj in analyzer_objs:
                preprocessed_file = analyzer_obj.preprocess()
                all_traces.append(preprocessed_file)

            # Creates global feature space for the current subdirectory
            for trace in all_traces:
                for element in trace:
                    if element not in feature_space:
                        feature_space[element] = index
                        index += 1

            # Appending the feature space and all traces of the current subdirectory to the lists
            global_feature_spaces.append(feature_space)
            all_traces_list.append(all_traces)

    # Returns a list of global feature spaces and a list of all trace lists
        print(filenames)
        return global_feature_spaces, all_traces_list
        
    # In this method, a Gaussian Mixture Model (GMM) is applied to classify phases in program traces. 
    # It uses Bayesian Information Criterion (BIC) to find the best model. 
    # This step is essential for identifying different phases in the program execution based on trace data, which is a key part of phase detection and feature selection.
    # def classify(self):
    #     feature_space,all_traces = self.create_training_data()
    #     obj = analyzer_class(None,feature_space,all_traces)
    #     # feature_vectors = obj.create_feature_vector_2()
    #     # feature_vectors = np.array(feature_vectors)
    #     # obj = analyzer_class("/Users/pranavpallavalli/Desktop/Thesis_analyzer/traces/trace.txt")
        
        
    #     # Create feature vectors for a particular time interval
    #     time = int(input("Please Enter a time interval: "))
    #     feature_vectors = obj.create_feature_vec_with_time(time)
    #     #List to Hold best prediction of phase changes
        # phases = []
        
        # #Loops through the feature vectors of each trace file and trains the GMM on each feature vector
        # # to classify into phases which aids in feature selection
        # for feature in feature_vectors:
        #     best_bic = inf
        #     best_phase = []
        #     feature = np.array(feature)
        #     print("resultant feature vector is:")
        #     print(feature)
        #     # Trying GMM with different numbers of components to find the best model based on BIC
        #     for n in range(2,len(feature)):
        #         gmm = GaussianMixture(n_components=n, random_state=12)
        #         gmm.fit(feature)
        #         bic = gmm.bic(feature)
        #         if bic < best_bic:
        #             best_bic = bic
        #             phase = gmm.predict(feature)
        #             best_phase = phase
        #     phases.append(list(best_phase))
    #     return phases,feature_vectors
    
    def classify(self):
        feature_space,all_traces = self.create_training_data()
        time = int(input("Please Enter a time interval: "))
        all_phases = []
        all_feature_vectors = []
        final_result = []
        for i in range(0,len(feature_space)):
            cur_feature_space = feature_space[i]
            cur_trace_files = all_traces[i]
            obj = analyzer_class(None,cur_feature_space,cur_trace_files)
            feature_vectors = obj.create_feature_vec_with_time(time)
            all_feature_vectors.append(feature_vectors)
            phases = []
            for feature in feature_vectors:
                best_bic = inf
                best_phase = []
                feature = np.array(feature)
                # print("resultant feature vector is:")
                # print(feature)
                # Trying GMM with different numbers of components to find the best model based on BIC
                for n in range(2,len(feature)):
                    gmm = GaussianMixture(n_components=n, random_state=12)
                    gmm.fit(feature)
                    bic = gmm.bic(feature)
                    if bic < best_bic:
                        best_bic = bic
                        phase = gmm.predict(feature)
                        best_phase = phase
                phases.append(list(best_phase))
            print("Original feature vectors for the program are: \n")
            print(feature_vectors)
            print("Original phases for the program are: \n")
            print(phases)
            
            groups = []
            index = 1
            for phase in phases:
                sub_arr = [index] * len(phase)
                groups.extend(sub_arr)
                index+=1
            print("Groups array is ----")
            print(groups)
            new_feature_vectors = self.correlation(phases,feature_vectors)
            print("Filtered feature vectors for the program based on feature selection with a target variable are \n")
            print(new_feature_vectors[0])
            reduced_feature_data_list = self.inter_feature_selection(new_feature_vectors[0])
            print("Feature vectors for program after further inter-feature selection are: \n")
            print(reduced_feature_data_list)
            best_n_components,best_gmm = self.perform_k_fold_cross_validation(feature_vectors,groups,(2,11),index-1)
            second_gmm_phases = self.train_second_gmm_on_filtered_features(reduced_feature_data_list,best_gmm)
            final_result.append(second_gmm_phases)
            print("Final Phase Change detection after passing filtered features through second GMM are: \n")
            print(second_gmm_phases)
        # return all_phases,all_feature_vectors        
        return final_result
            
            
    
    
    
    

    # This method calculates the correlation between features (event counts) and phases (target variables). 
    # It uses this correlation to perform feature selection, keeping only those features 
    # that have a significant correlation with the phase changes. 
    # This step helps in identifying which specific events (features) are most indicative of phase changes, 
    # thereby refining the feature set for more accurate phase detection.
    # The Pearson correlation coefficient measures the linear relationship between two variables or elements.
    # The Pearson correlation can help identify features that have a linear relationship with the occurrence of phase changes.
    def correlation(self, phases,feature_vectors):
        new_feature_vectors = []
        # Iterating over each trace files feature vector
        for i in range(0,len(feature_vectors)):
            feature = np.array(feature_vectors[i])
            phase = phases[i]
            if phases is None or len(phase)==0:
                new_feature_vectors.append([])
                continue
            target_vars = []
            prev = phase[0]
            # Creating target variables based on phase changes
            for j in range(0,len(phase)):
                ele = phase[j]
                if prev==ele:
                    target_vars.append(0)
                else:
                    target_vars.append(1)
                    prev = ele
            correlations = [] # List to store correlation values
            target_vars = np.array(target_vars)
            # Calculate pearson correlation for each feature based on target variables
            # For instance if we have feature vector - [[1,2],[3,4]] and target variables - [1,0]
            # where each index represents a particular event. So, index zero in in both arrays [1,2] and [2,4] correspond to the same event
            # For instance, in this case correlation between [1,3] and [1,0] and [2,4] and [1,0] is calculated
            # this methodology is used here
            for k in range(0,feature.shape[1]):
                corr = np.corrcoef(feature[:,k],target_vars)[0,1]
                correlations.append(corr)
            correlations = np.array(correlations)
            threshold = 0.5
            # Create a mask to select features based on correlation
            # Select only those features that meet the threshold of 0.5 and have a valid correlation value
            relevant_features_mask = (~np.isnan(correlations)) & (np.abs(correlations) >= threshold)
            relevant_features_mask = np.array(relevant_features_mask)
            # print(f"Phase is - {phase}")
            # print(f"Feature is: {feature}")
            # print(f"Type of mask is - {type(relevant_features_mask[0])}")
            # print(f"Feature maks is: {relevant_features_mask}")
            filtered_feature_vectors = feature[:, relevant_features_mask]
            # print(f"Filtered features are: {filtered_feature_vectors}")
            new_feature_vectors.append(filtered_feature_vectors)
        return new_feature_vectors,feature_vectors
    
    # def correlation(self,phases,feature_vectors):
    #     new_feature_vectors = []
    #     for i in range(0,len(phases)):
    #         cur_feature_vec = feature_vectors[i]
    #         cur_phases = phases[i]
    #         for j in range(0,len(cur_feature_vec)):
    #             for k in range(0,len(cur_feature_vec[j])):
    #                 feature_vector = np.array(cur_feature_vec[j][k])
    #                 phase = cur_phases[j][k]
    #                 target_vars = []
    #                 prev = phase[0]
    #         # Creating target variables based on phase changes
    #         for j in range(0,len(phase)):
    #             ele = phase[j]
    #             if prev==ele:
    #                 target_vars.append(0)
    #             else:
    #                 target_vars.append(1)
    #                 prev = ele
    #         correlations = [] # List to store correlation values
    #         target_vars = np.array(target_vars)
                    
                    
        
        
                   
    def train_second_gmm_on_filtered_features(self, filtered_feature_vectors,best_gmm):
        second_gmm_phases = []

        for filtered_feature in filtered_feature_vectors:
        
            filtered_feature = np.array(filtered_feature)
            
            # gmm = GaussianMixture(n_components=best_n_components, random_state=12)
            gmm = best_gmm
            gmm.fit(filtered_feature)
            phase = gmm.predict(filtered_feature)
            best_phase = phase
            if best_phase is None or len(best_phase)==0:
                second_gmm_phases.append([])
                continue
            prev = best_phase[0]
            
            target_vars = []
            # Creating target variables based on phase changes
            for j in range(0,len(phase)):
                ele = phase[j]
                if prev==ele:
                    target_vars.append(0)
                else:
                    target_vars.append(1)
                    prev = ele
            target_vars = np.array(target_vars)
            second_gmm_phases.append(list(target_vars))
        return second_gmm_phases
    
    
    def inter_feature_selection(self, feature_data_list):
        reduced_feature_data_list = []
        for feature_data in feature_data_list:
            if feature_data is None or len(feature_data)==0:
                reduced_feature_data_list.append(feature_data)
                continue
            if feature_data.shape[1] == 1:
                # If there's only one feature, just add it to the reduced list as is.
                reduced_feature_data_list.append(feature_data)
                continue
            
            remove_indices = set()
            n_features = feature_data.shape[1]
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    correlation, _ = pearsonr(feature_data[:, i], feature_data[:, j])
                    if abs(correlation) > 0.8:  # Using 0.8 as the correlation threshold
                        # Mark the feature with the higher index for removal
                        remove_indices.add(max(i, j))
            
            keep_indices = [i for i in range(n_features) if i not in remove_indices]
            reduced_feature_data = feature_data[:, keep_indices]
            reduced_feature_data_list.append(reduced_feature_data)
        
        return reduced_feature_data_list
    
    
    
    def perform_k_fold_cross_validation(self, data, groups, n_components_range, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        scores = []
        data = np.vstack(data)
        splits = gkf.split(data, groups=groups)
        for train_index, test_index in gkf.split(data, groups=groups):
            X_train, X_test = data[train_index], data[test_index]
            best_bic = np.inf
            best_n_components = None
            best_gmm = None

            for n_components in n_components_range:
                try:
                    gmm = GaussianMixture(n_components=n_components, random_state=12)
                    gmm.fit(X_train)
                    bic = gmm.bic(X_test)

                    if bic < best_bic:
                        best_bic = bic
                        best_n_components = n_components
                        best_gmm = gmm
                except Exception:
                    continue

            scores.append(best_bic)
            print(f"Best BIC: {best_bic} with {best_n_components} components")

        return best_n_components,best_gmm

    
    
    
    