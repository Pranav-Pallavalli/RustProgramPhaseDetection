import re

class analyzer_class:
    eventMappings = set()
    eventIndexMappings = dict()
    fileName = ""
    
    def __init__(self,file) -> None:
        self.eventMappings = {"jne","jge","jle","jmp"}
        self.eventIndexMappings = {}
        self.fileName = file
        self.unique_elements = set()
        self.globalFeatureIndexMapping = dict()
        
    
    def __init__(self,file,featureSpace = dict(),all_traces=[]) -> None:
        self.eventMappings = {"jne","jge","jle","jmp"}
        self.eventIndexMappings = {}
        self.fileName = file
        self.unique_elements = set()
        self.globalFeatureIndexMapping = featureSpace
        self.all_traces = all_traces
    
    def read_file(self):
        lines = []
        with open(self.fileName,"r") as file:
            i = 0
            for line in file:
                lines.append(line.strip())
        lines = [s for s in lines if s]
        for line in lines:
            if "No jmp" in line:
                lines[i] = line
                i+=1
                continue
            for ele in self.eventMappings:
                pos = line.find(ele)
                if pos!=-1:
                    line = line[pos:]
                    lines[i] = line
                    break
            i+=1
        return lines
    
    
    def preprocess(self):
        new_lines = []
        lines = self.read_file()
        i = 0
        while i<len(lines)-1:
            line = lines[i]
            next_line = lines[i+1]
            if next_line == "No jmp":
                line+=" false"
                i+=2
            else:
                line+=" true"
                i+=1
            new_lines.append(line)   
        new_lines.append(lines[len(lines)-1] + " true")
        return new_lines
    
    
    def find_unique_elements(self,user_input,lines):
        lines_output = []
        cur_line = []
        local_uniq_elements = set()
        for line in lines:
            if len(local_uniq_elements)>=user_input:
                local_uniq_elements = set()
                lines_output.append(cur_line)
                cur_line = []
            local_uniq_elements.add(line)
            self.unique_elements.add(line)
            cur_line.append(line)
        if len(cur_line)>0:
            lines_output.append(cur_line)
        return lines_output
            
            
    def create_feature_vec(self):
        lines = self.preprocess()
        user_input = int(input("Please Enter a time interval: "))
        lines = self.find_unique_elements(user_input,lines)
        feature_vec = []
        index_no = 0
        for line in lines:
            timestamp_vec = [0] * len(self.unique_elements)
            for statement in line:
                if statement not in self.eventIndexMappings:
                    self.eventIndexMappings[statement] = index_no
                    index_no+=1
                timestamp_vec[self.eventIndexMappings[statement]]+=1
            feature_vec.append(timestamp_vec)
        print(self.eventIndexMappings)
        return feature_vec
    
    def create_feature_vector_2(self):
        feature_vec = []
        for line in self.all_traces:
            timestamp_vec = [0] * len(self.globalFeatureIndexMapping)
            for statement in line:
                timestamp_vec[self.globalFeatureIndexMapping[statement]]+=1
            feature_vec.append(timestamp_vec)
        return feature_vec
    

    def create_feature_vec_with_time(self,time):
        feature_vec = []
        user_input = time
        for line in self.all_traces:
            full_features = []
            timestamp_vec = [0] * len(self.globalFeatureIndexMapping)
            local_set = set()
            for statement in line:
                if len(local_set)>=user_input:
                    full_features.append(timestamp_vec.copy())
                    local_set = set()
                    timestamp_vec = [0] * len(self.globalFeatureIndexMapping)
                local_set.add(statement)
                timestamp_vec[self.globalFeatureIndexMapping[statement]]+=1
            if len(timestamp_vec)>0:
                full_features.append(timestamp_vec.copy())
            feature_vec.append(full_features)
        return feature_vec
    
    