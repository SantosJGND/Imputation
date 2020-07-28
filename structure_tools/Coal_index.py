import math
import numpy as np
import itertools as it
import collections

def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)


import time
from sklearn.metrics import pairwise_distances



def get_config(dataw,nsamp):
    hap_str= [''.join([str(c) for c in x]).strip() for x in dataw]
    hap_str= {z:[x for x in range(nsamp) if hap_str[x] == z] for z in set(hap_str)}
    
    class_len= np.array([len(hap_str[z]) for z in hap_str.keys()])
    
    config= [sum(class_len == x) for x in range(1,nsamp + 1)]
    return config, hap_str



def process_array(dataT, return_config= False):
    '''
    Prepare haplotype array for coalescence inference algorithms.
    see: https://nbviewer.jupyter.org/github/SantosJGND/Coalescent/blob/master/Models_coalescence.ipynb
    
    '''
    nsamp= dataT.shape[0]

    config_dataw, hap_str= get_config(dataT,nsamp)

    hap_sol= list(hap_str.keys())
    hap_sun= np.array([np.array(list(x),dtype= int) for x in hap_sol])

    hap_size= [len(hap_str[x]) for x in hap_sol]
    hap_size= {z:[x for x in range(len(hap_size)) if hap_size[x] == z] for z in list(set(hap_size))}



    passing= hap_size.keys()
    pack= list(it.chain(*[hap_size[x] for x in passing]))
    passport= list(it.chain(*[[x]*len(hap_size[x]) for x in passing]))

    pack= [[pack[x],passport[x]] for x in range(len(pack))]
    pack= sorted(pack)
    pack= np.array(pack)

    Dict_mat= {0: 
               {
                   -2: hap_sun,
                   -1: [0] * hap_sun.shape[0],
                   0: pack
                  }
              }

    point_up= recursively_default_dict()
    
    if return_config:
        return Dict_mat, point_up, config_dataw

    else:
        return Dict_mat, point_up



def Inf_sites(Dict_mat,point_up,layer_range= 10, print_layer= True,print_time= True,sub_sample= 0, poppit= False):
    
    t1 = time.time()
   
    MRCA= False
    
    layer= 0
    
    
    for layer in range(layer_range):
        
        if MRCA:
            continue
        
        if print_layer:
            print('layer: {}; len: {}'.format(layer,len(Dict_mat[layer])-1))
        
        if len(Dict_mat[layer]) == 2:
            stdlone= max(Dict_mat[layer].keys())
            if sum(Dict_mat[layer][stdlone][:,1]) == 1:
                MRCA = True
                continue

        if poppit:
            if layer > 1:
                Dict_mat.pop(layer - 1)
            
        hap= list(Dict_mat[layer][-2])
        hap_coord= {}
        
        point_up[layer]= []
        
        Dict_mat[layer + 1]= {   
        }
        
        Quasi= []
        nodes= []
        new_haps= []
        
        keys_get= list(Dict_mat[layer].keys())
        
        if sub_sample:
            keys_get= np.random.choice(keys_get,sub_sample)
        
        for desc in keys_get:
            
            if desc >= 0:
                
                packet= list(Dict_mat[layer][desc])
                packet= np.array(packet)

                pack_above= [x for x in range(packet.shape[0]) if packet[x,1] > 1]
                pack_below= [x for x in range(packet.shape[0]) if packet[x,1] == 1]
                
                new_entries= np.array(list(range(len(pack_above)))) + len(Dict_mat[layer + 1])
                
                for nn in range(len(new_entries)):
                    tn= new_entries[nn]
                    while tn in nodes: 
                        tn += 1
                    
                    new_entries[nn]= tn
                
                who_loses= []
                
                ### Coalescence
                for z in range(len(pack_above)):
                    
                    who_loses.append(packet[pack_above[z],0])
                    
                    pack_synth= list(packet)
                    pack_synth= np.array(pack_synth)

                    pack_synth[pack_above[z],1] -= 1
                    
                    pack_tuple= sorted([tuple(x) for x in pack_synth])
                    
                    Query= [pack_tuple == x for x in Quasi]
                    Query= np.array(Query,dtype= int)
                    Query= np.where(Query == 1)[0] ## check if this changes anything
                    
                    if len(Query):
                        new_entries[z] = nodes[Query[0]]
                        
                    else:
                        pack_synth= np.array([list(x) for x in pack_tuple])
                        
                        pack_synth= pack_synth[pack_synth[:,1] > 0]
                        Dict_mat[layer + 1][new_entries[z]]= pack_synth
                        Quasi.append(pack_tuple)
                        nodes.append(new_entries[z])
                                
                packet_mob= packet[pack_above,:]
                
                packet_mob[:,1]= 1
                packet_mob[:,0]= who_loses
                packet_mob= np.hstack((np.zeros((packet_mob.shape[0], 1), dtype=packet_mob.dtype),packet_mob))
                packet_mob= np.hstack((packet_mob,np.zeros((packet_mob.shape[0], 1), dtype=packet_mob.dtype)))
                packet_mob[:,3] = -1 #######
                packet_mob[:,0]= new_entries
                packet_mob= np.hstack((np.zeros((packet_mob.shape[0], 1), dtype=packet_mob.dtype),packet_mob))
                packet_mob[:,0]= desc
                
                for y in packet_mob:
                    point_up[layer].append(y)
                                
                ## muts that can be removed. Assume mutations happen only once.
                exist= np.array(packet)[:,0]
                exist= np.array(hap)[exist,:]
                single= np.sum(exist,axis= 0)
                single= np.where(single==1)[0]
                ##
                    
                for edan in pack_below:
                    #
                    seq= hap[packet[edan,0]]
                    
                    #print(seq)
                    who= np.where(seq == 1)[0]
                    
                    who= [x for x in who if x in single]
                    
                    if len(who) == 0:
                        continue
                    
                                        
                    for mut in who:
                        
                        tribia= list(seq)
                        tribia= np.array(tribia)
                        tribia[mut]= 0

                        calc= pairwise_distances(np.array(tribia).reshape(1,-1), hap,
                                                        metric='hamming')[0]
                        
                        match= [x for x in range(len(calc)) if calc[x] == 0] 
                        
                        if len(match):
                            #print(match)
                                                        
                            for cl in match:
                                
                                pack_synth= list(Dict_mat[layer][desc])
                                pack_synth= np.array(pack_synth)
                                pack_synth[edan,1] -= 1
                                pack_synth= pack_synth[pack_synth[:,1] > 0]
                                
                                if cl in pack_synth[:,0]:
                                    cl_idx= list(pack_synth[:,0]).index(cl)
                                    pack_synth[cl_idx,1] += 1
                                    
                                else:
                                    new_row= np.array([cl,1])
                                    pack_synth= np.vstack((pack_synth,new_row.reshape(1,-1)))
                                
                                #### make function Query existant
                                new_entry= len(Dict_mat[layer + 1])
                                
                                while new_entry in Dict_mat[layer + 1].keys():
                                    new_entry += 1
                                
                                ###
                                path_find= 0 #########
                                pack_tuple= sorted([tuple(x) for x in pack_synth])

                                Query= [pack_tuple == x for x in Quasi]
                                Query= np.array(Query,dtype= int)
                                Query= np.where(Query == 1)[0] ## check if this changes anything

                                if len(Query):
                                    new_entry= nodes[Query[0]]

                                else:
                                    #print(pack_synth)
                                    pack_synth= np.array([list(x) for x in pack_tuple])
                                    Dict_mat[layer + 1][new_entry]= pack_synth
                                    Quasi.append(pack_tuple)
                                    nodes.append(new_entry)
                                ### 

                                point_up[layer].append([desc,new_entry,cl,path_find,mut]) ############
                        
                        else:
                            #
                            if len(new_haps):
                                #
                                calc= pairwise_distances(np.array(tribia).reshape(1,-1), np.array(new_haps),
                                                                                        metric='hamming')[0]
                                
                                match= [x for x in range(len(calc)) if calc[x] == 0]
                                
                                if len(match):
                                    
                                    new_idx= len(hap) + match[0]
                                
                                else:
                                    new_haps.append(tribia)
                                    new_idx= len(hap) + len(new_haps) - 1
                            
                            else:
                                new_haps.append(tribia)
                                new_idx= len(hap)
                            
                            #
                            pack_synth= list(Dict_mat[layer][desc])
                            pack_synth.append([new_idx,1]) # pack_synth.append([len(pack_synth),1])
                            pack_synth= np.array(pack_synth)
                            pack_synth[edan,1] -= 1
                            pack_synth= pack_synth[pack_synth[:,1] > 0]
                            
                            #### make function Query existant
                            new_entry= len(Dict_mat[layer + 1])
                            while new_entry in Dict_mat[layer + 1].keys():
                                new_entry += 1
                            
                            ###
                            path_find= 0 #########
                            pack_tuple= sorted([tuple(x) for x in pack_synth])

                            Query= [pack_tuple == x for x in Quasi]
                            Query= np.array(Query,dtype= int)
                            Query= np.where(Query == 1)[0] ## check if this changes anything

                            if len(Query):
                                new_entry = nodes[Query[0]]

                            else:
                                
                                pack_synth= np.array([list(x) for x in pack_tuple])
                                Dict_mat[layer + 1][new_entry]= pack_synth
                                Quasi.append(pack_tuple)
                                nodes.append(new_entry)

                            ####
                            point_up[layer].append([desc,new_entry,new_idx,path_find,mut])
        
        if new_haps:
            
            hap.extend(new_haps)
        
        point_up[layer]= np.array(point_up[layer])
        Dict_mat[layer + 1][-2] = np.array(hap)
        
        layer += 1
    
    t2 = time.time()
    tscale= 's'
    tpass= t2 - t1
    
    if tpass > 600:
        tpass = tpass / 60
        tscale= 'm'
    
    tpass= round(tpass,3)
    if print_time:
    
        print('time elapsed: {} {}'.format(tpass,tscale))
    
    return Dict_mat, point_up



############
############ get times

def theta_time(theta_blocks,max_time,Ngaps):

    time_breaks= np.linspace(1,max_time,Ngaps)

    theta_array= [
        time_breaks,
        theta_blocks
    ]

    theta_array= np.array(theta_array).T
    return theta_array


def theta_function(tnow, theta_time_array= []):
    
    if len(theta_time_array) == 0:
        print('no theta_time_array.')
        return tnow
    
    prime= np.where(theta_time_array[:,0] > tnow)[0]
    
    if len(prime) == 0:
        prime= theta_time_array[-1,1]
        return prime
    
    prime= theta_time_array[prime[0],1]
    
    return prime


def tree_ascent_times(root_lib,point_up,sink,init= [0],mu= 9e-8,theta_time_array= []):
    from structure_tools.Coal_probab import prob_coal, prob_mut
    from scipy.special import comb
    
    edge_weights= recursively_default_dict()
    paths= recursively_default_dict()
    paths_where= recursively_default_dict()
    node_times= recursively_default_dict()
    paths_time= recursively_default_dict()
    
    layer_nodes= {}
    
    if len(theta_time_array) == 0:
        print('no time array')
    
    for layer in list(range(sink)):
        
        where_to= point_up[layer]
        
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
                node_times[layer][desc]= [1]
            
            paths_ref= paths_where[layer][desc]
            paths_time_ref= node_times[layer][desc][0]
            Theta= theta_function(paths_time_ref, theta_time_array= theta_time_array)
            #
            
            Nt= Theta / (mu * 4)
            
            #
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
                    #
                    new_paths= list(paths_ref)
                    #          
                    new_paths.extend(new_paths)
                
                else:
                    new_paths= list(paths_ref)
                    #
                    new_paths.extend(new_paths[1:])
                
                going= point_up_house[row,1]
                
                ###
                packet= list(root_lib[layer+1][going])
                packet= np.array(packet)
                
                ###
                reff= list(root_lib[layer][desc])
                reff= np.array(reff)
                nsampn= sum(reff[:,1]) ### Samp this AC
                nsamp= sum(packet[:,1]) ## Samp next AC
                #print(nsampn)
                
                ### 
                now_time= node_times[layer][desc]
                
                mut_prob= prob_mut(Theta,nsampn)
                coal_prob= prob_coal(Theta,nsampn)

                prob_vec= [mut_prob,coal_prob]
                
                ### mut proportion
                
                reff_sing= reff[reff[:,1] == 1]
                
                mut_prop= reff_sing.shape[0] / reff.shape[0]
                
                ####
                if point_up_house[row,3]== 1:
                    
                    ave_time_coal= Nt / comb(nsampn, 2, exact=True)
                    ave_tme= [0,ave_time_coal]
                    
                    prob_split= (hap_split[0,1]) / nsamp
                
                else:
                    
                    ave_time_mut= 1 / (2 * Nt * mu)
                    ave_tme= [ave_time_mut,0]
                    prob_split= mut_prop
                
                ####
                ####
                pack= [x * prob_vec[point_up_house[row,3]] * prob_split for x in pack]
                
                ##
                new_paths= [round(ave_tme[point_up_house[row,3]],4) for x in new_paths]
                ##
                
                if going not in paths_where[layer + 1].keys():
                    paths_where[layer + 1][going]= pack
                    node_times[layer + 1][going]= new_paths
                
                else:
                    paths_where[layer + 1][going].extend(pack)
                    node_times[layer + 1][going].extend(new_paths)
        
        if len(edge_weights) == 0:
            edge_weights[sink][0] = 1

        for desc in paths_where[layer + 1].keys():
            edge_weights[layer + 1][desc]= sum(paths_where[layer + 1][desc])
            paths_where[layer + 1][desc]= [sum(paths_where[layer + 1][desc])]
            node_times[layer + 1][desc]= list(set(node_times[layer + 1][desc]))
            
    return edge_weights, paths_where, node_times



