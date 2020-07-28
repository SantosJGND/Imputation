import math
import numpy as np

import collections

def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)



def get_sinks(mrca,root_lib,point_up):
    
    sinks= []
    
    sink= max(root_lib.keys())
    
    if 0 not in root_lib[sink].keys():
        while 0 not in root_lib[sink].keys():
            sink -= 1
    
    #sink= (sink,0)
    found= 0
    
    while not found:
        is_it= np.array([np.array_equal(root_lib[sink][-2][x],mrca) for x in range(root_lib[sink][-2].shape[0])],dtype= int)
        which= np.where(is_it == 1)[0][0]
            
        AC_present= [x for x in root_lib[sink].keys() if x >= 0]
        AC_present= [x for x in AC_present if which in root_lib[sink][x][:,0]]
        
        if len(AC_present):
            
            sinks.extend(AC_present)
            
            found += 1
        
        else:
            sink -= 1
    
    if not len(sinks):
        mrca_str= ''.join([str(x) for x in mrca])
        print('ancestor {} not found'.format(mrca_str))
    
    return sink, sinks


#########
#########


def tree_descent(root_lib,point_up,sink,init= [0],Theta= 1):
    from structure_tools.Coal_probab import prob_mut, prob_coal
    
    edge_weights= recursively_default_dict()
    paths= recursively_default_dict()
    paths_where= recursively_default_dict()
    
    for layer in list(range(1,sink + 1))[::-1]:
        
        #print('layer: {}'.format(layer))
        where_to= point_up[layer - 1]
        
        if layer == sink:
            starters= init
        else:
            starters= list(set(where_to[:,1]))
        #print(where_to)
        
        for desc in list(set(where_to[:,1])):
            
            ###
            #print(packet)
            ###
            
            point_up_house= where_to[where_to[:,1] == desc]
    
            #point_up[layer].extend(point_up_house)
            #point_up_house= np.array(point_up_house)

            if not len(paths):

                paths= {
                    0: 1
                }

                paths_where[layer][desc]= [1]


            for row in range(point_up_house.shape[0]):
                pack= list(paths_where[layer][desc])

                here= desc #point_up_house[row,1]
                node_next= np.array(root_lib[layer][here])

                who_lost= point_up_house[row,2] # hap set that originates the mutation / coalescent event
                hap_split= node_next[node_next[:,0] == who_lost] # hap set row
                
                if row > 0:
                    new_entry= len(paths)
                    while new_entry in paths.keys():
                        new_entry += 1
                
                going= point_up_house[row,0]
                
                ###
                packet= list(root_lib[layer-1][going])
                packet= np.array(packet)
                nsamp= sum(packet[:,1]) ## Samp next AC
                
                mut_prob= prob_mut(Theta,nsamp)
                coal_prob= prob_coal(Theta,nsamp)

                prob_vec= [mut_prob,coal_prob]
                
                ### mut proportion
                reff= root_lib[layer-1][going]
                reff_sing= reff[reff[:,1] == 1]
                
                mut_prop= reff_sing.shape[0] / reff.shape[0]
                
                ### This makes the difference.              
                nsampn= sum(root_lib[layer][desc][:,1]) ### Samp this AC
                
                ####
                if point_up_house[row,3]== 1:
                    prob_split= (hap_split[0,1]) / nsampn
                
                else:
                    prob_split= mut_prop
                
                pack= [x * prob_vec[point_up_house[row,3]] * prob_split for x in pack]

                if going not in paths_where[layer - 1].keys():
                    paths_where[layer - 1][going]= pack

                else:
                    paths_where[layer - 1][going].extend(pack)



        if len(edge_weights) == 0:
            edge_weights[sink][0] = 1

        for desc in paths_where[layer - 1].keys():
            edge_weights[layer - 1][desc]= sum(paths_where[layer - 1][desc])
            paths_where[layer - 1][desc]= [sum(paths_where[layer - 1][desc])]
            

    return edge_weights, paths_where
    


def Descent_return(point_up,root_lib,layer=0,start=0,Theta= 1,prob_vec= []):
    
    
    sink= max(root_lib.keys())
    
    if 0 not in root_lib[sink].keys():
        while 0 not in root_lib[sink].keys():
            sink -= 1
    
    
    
    node_weigths, paths_reverse = tree_descent(root_lib,point_up,sink,Theta= Theta)
       
    #return paths_reverse
    return [node_weigths[0][0]]



############ Tree descent for haps
############

from scipy.special import comb

def tree_descent_gen(root_lib,point_up,sink,init= [0],Theta= 1,Nt= 1000,mu= 9e-8):
    
    from structure_tools.Coal_probab import prob_coal, prob_mut
    edge_weights= recursively_default_dict()
    paths= recursively_default_dict()
    paths_where= recursively_default_dict()
    node_bins= recursively_default_dict()
    
    for layer in list(range(1,sink + 1))[::-1]:
        
        #
        where_to= point_up[layer - 1]
        
        if layer == sink:
            starters= init
        else:
            starters= list(set(where_to[:,1]))
        
        for desc in list(set(where_to[:,1])):
            
            ###
            #print(packet)
            ###
            
            point_up_house= where_to[where_to[:,1] == desc]
            
            if not len(paths):
                paths= {
                    0: 1
                }
                ## this will store time per path
                paths_vector= [1]

                paths_where[layer][desc]= [0]
                node_bins[layer][desc]= 1
            
            paths_ref= paths_where[layer][desc]
            
            for row in range(point_up_house.shape[0]):
                pack= list(paths_where[layer][desc])

                here= desc #point_up_house[row,1]
                node_next= np.array(root_lib[layer][here])

                who_lost= point_up_house[row,2] # hap set that originates the mutation / coalescent event
                hap_split= node_next[node_next[:,0] == who_lost] # hap set row
                
                ### paths attention
                
                if row > 0:
                    new_entry= len(paths)
                    while new_entry in paths.keys():
                        new_entry += 1
                    #
                    new_paths= list(paths_ref)
                    #          
                    new_paths.extend(new_paths)
                
                else:
                    new_paths= list(paths_ref)
                    #
                    new_paths.extend(new_paths[1:])
                
                
                going= point_up_house[row,0]
                
                ###
                packet= list(root_lib[layer-1][going])
                packet= np.array(packet)
                nsamp= sum(packet[:,1]) ## Samp next AC
                
                mut_prob= prob_mut(Theta,nsamp)
                coal_prob= prob_coal(Theta,nsamp)

                prob_vec= [mut_prob,coal_prob]
                
                ### mut proportion
                reff= root_lib[layer-1][going]
                reff_sing= reff[reff[:,1] == 1]
                
                mut_prop= reff_sing.shape[0] / reff.shape[0]
                
                ###
                ###
                nsampn= sum(root_lib[layer][desc][:,1]) ### Samp this AC
                
                ####
                if point_up_house[row,3]== 1:
                    ### Ave_time
                    ave_time_coal= Nt / comb(nsamp, 2, exact=True)
                    ave_tme= [0,ave_time_coal]
                    
                    prob_split= (hap_split[0,1]) / nsampn
                
                else:
                    ave_time_mut= 1 / (2 * Nt * mu)
                    ave_tme= [ave_time_mut,0]
                    prob_split= mut_prop
                
                ###
                pack= [x * prob_vec[point_up_house[row,3]] * prob_split for x in pack]
                
                ### path times
                new_paths= [x + ave_tme[point_up_house[row,3]] for x in new_paths]
                
                if going not in paths_where[layer - 1].keys():
                    paths_where[layer - 1][going]= new_paths
                
                else:
                    paths_where[layer - 1][going].extend(new_paths)
                    
                ###
        
        
        for desc in paths_where[layer - 1].keys():
            if len(paths_where[layer - 1][desc]) >100:
                paths_where[layer - 1][desc]= list(set(paths_where[layer - 1][desc]))
            

    return edge_weights, paths_where, node_bins, paths_vector
    

