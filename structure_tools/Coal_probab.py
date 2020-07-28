import math
import numpy as np
import itertools as it

import collections

def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)


def prob_coal(theta,nsamp):
    
    p= (nsamp - 1) / (nsamp - 1 + theta)
    
    return p

def prob_mut(theta,nsamp):
    
    p= theta / (nsamp - 1 + theta)
    
    return p

###

def Ewens_exact(config_data,theta):
    
    n_samp= sum([(x + 1) * config_data[x] for x in range(len(config_data))])
    
    ThetaN= [theta + x for x in range(len(config_data))]
    ThetaN0= 1
    for y in ThetaN:
        ThetaN0 = ThetaN0 * y
    
    factor_front= math.factorial(len(config_data)) / ThetaN0
    
    classes= 1
    
    for j in range(len(config_data)):
        comb= theta / (j+1)
        comb= comb**config_data[j]
        
        prod= comb * (1 / math.factorial(config_data[j]))
        
        classes= classes * prod
    
    return factor_front * classes

####

def Ewens_recurs(config_vec,theta,prob_array,Pin,prob_bound = 1):
    n_samp= sum([(x + 1) * config_vec[x] for x in range(len(config_vec))])
    
    if config_vec == [1]:
        ## boundary
        
        prob_left= Pin * prob_bound
        
        prob_array.append(prob_left)
        
        return prob_array
    
    if config_vec[0] > 0:
        ## mutation
        prob_left= prob_mut(theta,n_samp)
        
        new_conf= list(config_vec)[:(n_samp - 1)]
        new_conf[0]= new_conf[0] - 1
        
        prob_next= Pin * prob_left
        
        Ewens_recurs(new_conf,theta,prob_array,prob_next)
    
    
    if sum(config_vec[1:]) > 0:
        ## coalesc
        prob_right_I = prob_coal(theta,n_samp)
        
        jsel= [x for x in range(1,len(config_vec)) if config_vec[x] > 0]
        
        for sub in jsel:
            ##  coalesce for all classes still holding more than one allele.
            
            jprop= sub * (config_vec[sub - 1] + 1) / (n_samp - 1)
            
            new_conf= list(config_vec)
            new_conf[sub] -= 1
            new_conf[sub - 1] += 1
            new_conf= new_conf[:(n_samp - 1)]
            
            prob_right= prob_right_I * jprop

            prob_next= Pin * prob_right

            Ewens_recurs(new_conf,theta,prob_array,prob_next)
    
    return prob_array


#######
####### Run up unique - removing DCGs



def runUp_unique(Up_lib,Root,layer=0,start= 0,ori= [],store_unique= {},Theta= 1,probs= [],prob_vec= [],Pin= 1):
    
    if not len(Up_lib[layer]):
        
        if ori not in store_unique.keys():
            
            end_story= (layer,start)
            
            store_unique[ori][end_story]= Pin
            

        return store_unique
    
    Ways_all= list(Up_lib[layer])
    Ways_all= np.array(Ways_all)
    Ways= Ways_all[Ways_all[:,0] == start]    
    
    for row in range(Ways.shape[0]):
        ori_here= tuple(ori)
        action= Ways[row,3]

        ## identifying the next node.
        next_stop= Ways[row,1]
        node_next= Root[layer + 1][next_stop]
        
        if len(ori) < 3:
            ori_here= (ori[0],ori[1],next_stop)
            
        #if ori_here in store_unique.keys():
        #    return store_unique

        ## calculate mut. and coal. probs based on next node sample size. 
        this_mat= Root[layer][start]
        nsamp= sum(this_mat[:,1])

        mut_prob= prob_mut(Theta,nsamp)
        coal_prob= prob_coal(Theta,nsamp)

        ### Mut was coded to 0, coalescence to 1.
        probs= [mut_prob,coal_prob]
        
        probe= probs[action] # edge = [mutation, coalescence] 

        ###

        who_lost= Ways[row,2] # hap set that originates the mutation / coalescent event
        hap_split= node_next[node_next[:,0] == who_lost] # hap set row
        

        if action == 1:
            # coalescence 
            prob_split= (hap_split[0,1]) / sum(node_next[:,1]) # proportion of ancestral hap set in previous AC

            probe= probe * prob_split

        if action == 0:
            # mutation
            
            singletons= this_mat[this_mat[:,1] == 1].shape[0]
            prob_split= singletons / this_mat.shape[0]
            #prob_split= (hap_split[0,1]) / sum(node_next[:,1])  # probability that this particular hap mutated.

            probe= probe * prob_split 
        
        ###

        new_pin= Pin * probe ## Probability inheritance.
        
        if Ways_all[Ways_all[:,1] == next_stop].shape[0] > 1:
            next_story= (layer + 1, next_stop)
            
            store_unique[ori_here][next_story]= new_pin
            
            runUp_unique(Up_lib,Root,layer=layer + 1,start=next_stop,ori=next_story,store_unique=store_unique,
                 Theta= Theta,probs= probs,prob_vec= prob_vec,Pin= 1)
        
        else:
        
            runUp_unique(Up_lib,Root,layer + 1,start= next_stop,ori=ori_here,store_unique=store_unique,
                         Theta= Theta,probs= probs,prob_vec= prob_vec,Pin= new_pin)
    
    return store_unique




def split_browse(browse_dict,permission= True):
    
    skeys= list(browse_dict.keys())
    
    dirs= list(it.chain(*[[x]*len(browse_dict[x]) for x in skeys]))
    
    sets= list(it.chain(*[browse_dict[x] for x in skeys]))
    
    first_lib= {
        z: {dirs[x]:browse_dict[dirs[x]][sets[x]] for x in range(len(sets)) if sets[x] == z} for z in list(set(sets))
    }
    
    ### compression step 
    ### get rid of cycles
    
    if permission:
        for cl in first_lib.keys():

            new_roots= list(first_lib[cl].keys())
            new_suf= [x[:2] for x in new_roots]

            new_seeds= {
                z: [x for x in range(len(new_suf)) if new_suf[x] == z] for z in list(set(new_suf))
            }
            
            new_seeds= {
                new_roots[new_seeds[z][0]]: sum([first_lib[cl][new_roots[x]] for x in new_seeds[z]]) for z in new_seeds.keys()
            }

            first_lib[cl]= new_seeds
    
    ###
    ###
        
    return first_lib



def rec_back(node,node_input,val_vec= [],Pin= 1):
    
    if node[:2] == (0,0):
        
        val_vec.append(Pin)
        return val_vec
    
    if node not in node_input.keys():
        print('node {} has no descendents.'.format(node))

    for desc in node_input[node].keys():
        
        new_desc= tuple(desc[:2])
        new_pin= Pin * node_input[node][desc]
        
        rec_back(new_desc,node_input,val_vec= val_vec,Pin=new_pin)

    return val_vec



def run_unique_combine(Up_lib= {},Root= {},layer= 0,start= 0,origin= [0,0],store_unique= {},
                       Theta= 1,probs= [],prob_vec= [],Pin= 1):
    
    store_unique= recursively_default_dict()
    
    Browse= runUp_unique(Up_lib= Up_lib,Root= Root,layer= layer,start= start,ori= origin,
                         store_unique= store_unique,Theta= Theta,probs= probs,prob_vec= prob_vec,Pin= Pin)
    
    Uni_ends = split_browse(Browse)
    sink= max(Uni_ends.keys())
    
    paths_forward= rec_back(sink,Uni_ends,val_vec= [],Pin= 1)
    
    return paths_forward


############## Faisal 2015
############## Tree Descent


def tree_descent(root_lib,point_up,sink,init= [0],Theta= 1):

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


#################### Faisal 2015
#################### tree_ascent


### Another take, backwards instead of forward in time.
###

def tree_ascent(root_lib,point_up,sink,init= [0],Theta= 1):
    
    edge_weights= recursively_default_dict()
    paths= recursively_default_dict()
    paths_where= recursively_default_dict()
    
    layer_nodes= {}
    
    for layer in list(range(sink)):
        
        where_to= point_up[layer]
        
        theta_now= [Theta]
        layer_nodes[layer]= []
        
        for desc in list(set(where_to[:,0])):
            
            ###
            #print(packet)
            ###
            
            point_up_house= where_to[where_to[:,0] == desc]
    
            if not len(paths):

                paths= {
                    0: 1
                }
                
                paths_where[layer][desc]= [1]
            
            for row in range(point_up_house.shape[0]):
                pack= list(paths_where[layer][desc])
    
                there= point_up_house[row,1]
                node_next= list(root_lib[layer + 1][there])
                node_next= np.array(node_next)
                
                who_lost= point_up_house[row,2] # hap set that originates the mutation / coalescent event
                hap_split= node_next[node_next[:,0] == who_lost] # hap set row
                #print(hap_split)
                
                if row > 0:
                    new_entry= len(paths)
                    while new_entry in paths.keys():
                        new_entry += 1
                
                going= point_up_house[row,1]
                
                ###
                packet= list(root_lib[layer+1][going])
                packet= np.array(packet)
                
                ###
                reff= list(root_lib[layer][desc])
                reff= np.array(reff)
                nsampn= sum(reff[:,1]) ### Samp this AC
                nsamp= sum(packet[:,1]) ## Samp next AC
                
                mut_prob= prob_mut(Theta,nsampn)
                coal_prob= prob_coal(Theta,nsampn)

                prob_vec= [mut_prob,coal_prob]
                
                ### mut proportion
                
                reff_sing= reff[reff[:,1] == 1]
                
                mut_prop= reff_sing.shape[0] / reff.shape[0]
                
                ####
                if point_up_house[row,3]== 1:
                    prob_split= (hap_split[0,1]) / nsamp
                
                else:
                    prob_split= mut_prop
                
                ####
                ####
                pack= [x * prob_vec[point_up_house[row,3]] * prob_split for x in pack]

                if going not in paths_where[layer + 1].keys():
                    paths_where[layer + 1][going]= pack

                else:
                    paths_where[layer + 1][going].extend(pack)

        if len(edge_weights) == 0:
            edge_weights[sink][0] = 1

        for desc in paths_where[layer + 1].keys():
            edge_weights[layer + 1][desc]= sum(paths_where[layer + 1][desc])
            paths_where[layer + 1][desc]= [sum(paths_where[layer + 1][desc])]
            

    return edge_weights, paths_where



def Ascent_return(point_up,root_lib,layer=0,start=0,Theta= 1,prob_vec= []):
    
    
    sink= max(root_lib.keys())
    
    if 0 not in root_lib[sink].keys():
        while 0 not in root_lib[sink].keys():
            sink -= 1
    
    
    
    node_weigths, paths_reverse = tree_ascent(root_lib,point_up,sink,Theta= Theta)
       
    #
    return [node_weigths[sink][0]]


