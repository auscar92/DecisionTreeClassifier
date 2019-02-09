# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:02:01 2019

@author: ausca
"""
import math
import numpy as np
import pandas as p

# =============================================================================
# def calcEntropy(att, att_name):
#     
#     counts = att[att_name].value_counts(normalize=True)
#     vals = counts.values
#     entrop = 0
#     for x in np.nditer(vals):
#     entrop -= x*math.log2(x)
#     return entrop
#  
# =============================================================================

class Node:
    def __init__(self, data):
        self.data = data
        self.children = {}
        self.isLeaf = False

    def getData(self):
        return self.data

    def getChildren(self):
        return self.children

    def setData(self,newdata):
        self.data = newdata

    def addChild(self,newnext, key):
        self.children[key] = newnext
    
    def getLeaf(self):
        return self.isLeaf
    
    def setLeaf(self, newLeaf):
        self.isLeaf = newLeaf
        
        
class decisTree:
    def __init__(self):
        self.root = None
        self.df_set = []
    
    def __getRoot__(self):
        return self.root
    
    def attribute_entrop(self, col_name):
        att_entrop = {}
        self.attributes.remove(col_name)
        for att_name in self.attributes:
            temp_df_set = []
            for df in self.df_set:
                groupeddf = df.groupby(att_name)
                tar_types = df[att_name].value_counts()
                for x in tar_types:
                    temp_df_set.append(groupeddf.get_group(x))
                
            total_entrop = 0
            for df_name in temp_df_set:
                counts = df_name[self.tar_name].value_counts(normalize=True)
                vals = counts.values
                
                entrop = 0
                for x in np.nditer(vals):
                    if x == 0:
                        entrop -= 0
                    else:
                        entrop -= x*math.log2(x)
                total_entrop += (df_name[self.tar_name].size/self.df_length)*entrop
            att_entrop[att_name] = total_entrop
        self.df_set = temp_df_set
        entrop_df = p.Series(att_entrop)
        lowest_entrop = entrop_df.idxmin
        lowest_entrop_val = entrop_df.min
        if lowest_entrop_val != 0:
            temp_node = Node(lowest_entrop)
            temp_node.setLeaf(False)
            att_vals = self.orig_df[lowest_entrop].value_counts()
            for x in att_vals:
                temp_node.addChild(self.attribute_entrop(lowest_entrop), x)
            return temp_node
        else:
            temp_node = Node(lowest_entrop)
            temp_node.setLeaf(True)
            return temp_node
            
        
    def make_tree(self, df, tar_name):
        self.df_set.append(df)
        self.attributes = df.columns.tolist()
        self.tar_name = tar_name
        self.df_length = df[tar_name].size
        self.orig_df = df
        self.root = self.attribute_entrop(tar_name)
       
    def test_tree(self, df):
        results = []
        for row in df.rows:
            results.append(row_val(row, self.root))
        return results
        
    def row_val(self, df_row, parent_node):
        if parent_node.isLeaf() == True:
            return parent_node.data
        else:
            for k,v in parent_node.getChildren():
                if df_row[parent_node.data] == k:
                    return self.row_val(df_row, v)
# =============================================================================
# Test array
# =============================================================================
#read in the student math class dataset
names = ['index', 'age', 'prescription', 'astigmatic', 'tear_rate', 'contacts']
s_data = p.read_csv("lenses.csv", names = names, sep=" ")

#gets rid of rows with null values in them
s_data = s_data.dropna(how='any', axis=0)

contact_df = s_data.drop(columns=['index'])

train_df = contact_df[:20]
test_df = contact_df[20:]

contact_tree = decisTree()
contact_tree.make_tree(train_df, 'contacts')
print(contact_tree.root.children)