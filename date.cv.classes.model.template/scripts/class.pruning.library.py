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
    elif relative=='full':
	return nodes[0:current+1];
    elif relative=='full_child':
	return nodes[0:current+2];

def WhoAreChildren(df):
    return list(set(df.unique()).difference([np.nan]))

def TaxonomyDataFrame(x):
        
    # Let's create a dataframe with all of the OTU in the data set and detaied taxonomic information
    
    taxonomy_se = [];
    for microbe in x.keys():
        taxonomy = microbe
        taxonomy = pd.Series(dict([clade.split('__') for clade in taxonomy.split(';')]),name = microbe)
        taxonomy_se.append(taxonomy)
        
    taxonomy_df = pd.DataFrame(taxonomy_se)
    taxonomy_df = taxonomy_df.reindex_axis(['k','p','c','o','f','g','s','otu'], axis=1);
    
    return taxonomy_df

def SegregateMicrobes(df):
    
    x_microbes = df.loc[:,[dfk.startswith('k__') for dfk in df.keys()]];
    x_clinical = df.loc[:,[not dfk.startswith('k__') for dfk in df.keys()]];
    
    return x_microbes, x_clinical 

def dropNonInformativeClades(x):
    
    nodes  = ['k','p','c','o','f','g','s'];
    toDrop = [];    
   
    # split input data_frame into microbes versus everything else; only prune on the microbes subset
    # how do you know a column corresponds to a microbe: its key begins with "k__"
    x_microbes,x_clinical = SegregateMicrobes(x);  
	 
    # Let's create a dataframe with all of the OTU in the data set and detaied taxonomic information
    taxonomy_df = TaxonomyDataFrame(x_microbes)

    # Let's try to parse through each otu and see if it has any relatives at above clades. If so, we will manually create a variable name for that shared clade. 
    for n in nodes[1:]:
	
	full_child   = GrabRelativeNode(n,'full_child');
	child        = GrabRelativeNode(n,'child');
	full         = GrabRelativeNode(n,'full');
	
	df           = taxonomy_df.loc[:,full_child];
	grouped_df   = df.groupby(full).groups;

	for parent,household in grouped_df.iteritems():
		parent_df = df.loc[household,child];
		if len(WhoAreChildren(parent_df))==1:
			toDropName = ';'.join([xx+'__'+yy for xx,yy in zip(full,parent)]);
			toDrop.append(toDropName);
		#endif
	#endif
    #endfor

    x_microbes.drop(labels=toDrop,axis=1,inplace=True)    
    x = x_microbes.join(x_clinical,how='left');
	
    return x
