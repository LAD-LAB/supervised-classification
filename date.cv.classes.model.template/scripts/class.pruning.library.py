#!/usr/bin/env python 
#
# ___author___ : Firas Said Midani
# ___e-mail___ : firas.midani@duke.edu
# ___date_____ : 2015.12.20
# ___version__ : 1.0
#
#
# LIST OF FUNCTIONS
# 
# GrabRelativeNode
# 	taxonomic ranks are {kingdom,phylum,class,order,family,genus,species,otu}
#   given a particular taxonomic rank, you can get itself, its parent, its daughter, 
#   or its upstream relatives back in the form of list of initials {k,p,c,o,f,g,s}
#
# WhoAreChildren
#   given a TaxonomyDataFrame (see below function), taxonomic rank, and name of clade
#   identify who are the children of this clade
# 
# HowManyChildren
#   given a TaxonomyDataFrame (see below function), taxonomic rank, and name of clade
#   identify how many  children does this clade have
#
# FullName
#   Return the full name of a clade beginning with kingdom
#
# WriteName
#   given a single row of a TaxonomyDataFrame, it returns the full name of the clade
#
# TaxonomyDataFrame
#   given a matrix of samples by bacterial OTU IDs, it constructs a data frame with
#   an index of the full taxonomic name (kingdom to OTU) in the GreenGenes format 
#   and columns of the taxonomic ranks for each index {k,p,c,o,f,g,s,otu}
# 
# dropNonInformativeClades
#   given a feature matrix and an OTU mapping dataframes
#   it identifies clades that are not informative (because their sole child clade contains
#   the exact same information, then drops those non informative clades 
#
# LIST OF CLASSES
#
# taxonomy
#	Given a taxonomy formatted using GreenGenes format, break down the taxonomy into ranks


import \
numpy  as np, \
pandas as pd, \
pprint as pp

class taxonomy(object):
    
    def __init__(self,name):
        self.taxonomy = name
    
    def breakdown(self):
        taxa      = self.taxonomy; 
        #taxa      = taxa.replace('_;','_na;')
        taxa_dict = dict([ii.split('__') for ii in taxa.split(';')])
        return taxa_dict
    
    def dictionary(self):
        record = self.breakdown();
        for k,v in record.iteritems():
            record[k]=[v]
        return record
        
def GrabRelativeNode(n,relative):
    nodes   = ['k','p','c','o','f','g','s','otu',''];
    current = np.where([n==node for node in nodes])[0][0];
    if   relative=='self':
         return nodes[current];
    elif relative=='parent':
         return nodes[current-1];
    elif relative=='child':
         return nodes[current+1];
    elif relative=='upstream':
         return nodes[0:current];

def WhoAreChildren(df,node,parent):
    child    = GrabRelativeNode(node,'child')
    children = set(df.loc[df.loc[:,node]==parent,child].unique())
    children = list(children.difference([np.nan]))
    return children

def HowManyChildren(df,node,parent):
    children = WhoAreChildren(df,node,parent);
    return len(children)
     
def FullName(df,node,self,which):
    names = GrabRelativeNode(node,'upstream');
    names = df.loc[df.loc[:,node]==self,names].iloc[0,:]
    if    which=='upstream':
          return WriteName(names)
    elif  which=='include':
          return WriteName(names)+';'+node+'__'+self
    
def WriteName(df):
    ranks = df.keys();
    names = df.values;
    return ";".join([r+'__'+n for r,n in zip(ranks,names)])
    

def TaxonomyDataFrame(x,otu_map):
        
    # Let's create a dataframe with all of the OTU in the data set and detaied taxonomic information
    
    taxonomy_se = [];
    for microbe in x.keys():
        if (isinstance(microbe,int)) | (str(microbe).isdigit()):
            microbe  = otu_map.loc[int(microbe),'taxonomy']+' ;otu__'+str(microbe);
        taxonomy = microbe
        taxonomy = taxonomy.replace(';  ',';');
        taxonomy = pd.Series(dict([clade.split('__') for clade in taxonomy.split(';')]),name = microbe)
        taxonomy_se.append(taxonomy)
        
    taxonomy_df = pd.DataFrame(taxonomy_se)
    taxonomy_df = taxonomy_df.reindex_axis(['k','p','c','o','f','g','s','otu'], axis=1);
    
    return taxonomy_df
    
def dropNonInformativeClades(x,otu_map):
    
    nodes  = ['k','p','c','o','f','g','s'];
    toDrop = [];    
    
    # Let's create a dataframe with all of the OTU in the data set and detaied taxonomic information
    taxonomy_df = TaxonomyDataFrame(x,otu_map)

    # Let's try to parse through each otu and see if it has any relatives at above clades. If so, we will manually create a variable name for that shared clade. 
    for n in nodes[1:]:
        for parent in list(set(taxonomy_df.loc[:,n].unique()).difference([np.nan])):
            if HowManyChildren(taxonomy_df,n,parent)==1:
                toDrop.append(FullName(taxonomy_df,GrabRelativeNode(n,'self'),parent,'include'))

    x.drop(labels=toDrop,axis=1,inplace=True)    
   
    return x