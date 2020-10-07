

import pandas as pd 
import numpy as np 

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import MeanShift, estimate_bandwidth

###
def clean_geno(genotype, nan_char= 9,
               het_char= 1):
    '''
    return indicies that are not empty or populated only by nan or het characters.
    '''
    gen_add= np.sum(genotype,axis= 0)
    gen_sum= gen_add != 0
    
    #
    nan_char= 9
    het_char= 1

    nan_geno= genotype == nan_char
    nan_geno= np.array(nan_geno,dtype= int)
    #
    het_char= genotype == het_char
    het_char= np.array(het_char,dtype= int)
    #
    arch= genotype > 0
    arch= np.array(arch,dtype= int)
    arch= np.sum(arch,axis= 0)
    
    nan_row= np.sum(nan_geno,axis= 0)
    het_row= np.sum(het_char,axis= 0)
    
    clob_sum= nan_row + het_row
    
    keep_pos= [x for x in range(genotype.shape[1]) if gen_sum[x] and arch[x] != nan_row[x] and arch[x] != het_row[x] and arch[x] != clob_sum[x]]
    
    return keep_pos


###
def window_parse(summary, centre= 0, wind_sizes= 20, wind_prox= 1e6):
    '''
    select window based on proximity.
    '''
    obs_pos= int(summary.POS[centre])
    wst= [x for x in range(summary.shape[0]) if abs(int(summary.POS[x]) - obs_pos) <= wind_prox and x != centre]
    #wst= [x for x in wst if x < (centre-int(wind_sizes/2)) or x > (centre+int(wind_sizes/2))]
    
    return wst


def lwind_extract(genotype, idx= 50, wind_sizes= 50, mask_pos= []):
    '''
    '''
    width= int(wind_sizes/2)
    lb= idx - width
    lb= [0,lb][int(lb > 0)]
    ub= idx + width
    ub= [ub,genotype.shape[1]-1][int(ub >= genotype.shape[1])]
    retain= []
    d= int(lb)

    while len(retain) < wind_sizes and d < ub:
        if d not in mask_pos:
            retain.append(d)
        d+= 1
    
    retain= sorted(retain)
    lwind= genotype[:,retain]
    
    return lwind


def lwind_extractv2(genotype, idx= 50, wind_sizes= 50, mask_pos= []):
    '''
    '''

    retain= [idx]
    ub= idx + 1
    lb= idx - 1
    
    while len(retain) < wind_sizes:

        if ub not in mask_pos:
            if ub < genotype.shape[1] and len(retain) < wind_sizes:
                retain.append(ub)
        ub += 1

        if lb not in mask_pos:
            if lb >= 0 and len(retain) < wind_sizes:
                retain.append(lb)
        lb -= 1
    
    retain= sorted(retain)
    lwind= genotype[:,retain]
    
    return lwind


def recover_hap(background,like_diet,pca_spec,
                scale= 1, round_t= True):
    '''
    recover haplotype using pca inverse transform,  
    '''
    tf_max= background[np.argmax(like_diet)]
    
    tf_rec= pca_spec.inverse_transform(tf_max.reshape(1,-1))

    if round_t:
        tf_rec[tf_rec <= 0] = 0
        tf_rec= np.round(tf_rec) * scale
        tf_rec= np.array(tf_rec,dtype= int)
    
    return tf_rec


###
###
###
from sklearn.metrics import pairwise_distances


def target_wdDist(genotype, keep_tools, wind_extract_func, avail_coords= [],
                  nan_obs= [0,0],
               wind_sizes= 100,
                Nrep= 400,
                ncomps= 5,
                 ind_min= 50,
                 int_check= 50,
                 avoid_range= 0):
    '''
    Given avail_coords list of possible features, 
    Parse every possible window for observations bearing codes to avoid using the function `code_check`.
    find the group of windows that maximizes the number of observations without any code to avoid and group size.
    method:
    - Use binary profiles of windows, indicating the absence of codes to avoid by observation.
    - Use PCA and meanshift clustering to find similar patterns of missingness, select group of windows minimizing missingness.
    '''
    pca = PCA(n_components=ncomps, whiten=False,svd_solver='randomized')
    
    ###
    if not avail_coords:
        avail_coords= list(range(genotype.shape[1]))
    
    nan_acc= nan_obs[0]
    nan_pos= nan_obs[1]
    other_obs= [x for x in range(genotype.shape[0]) if x != nan_acc]
    
    ###
    nan_array= []
    Seq_store= {}
    select_same= []
    dist_store= []
    already_visited= [0] * len(avail_coords)
    
    d= 0
    trail= []

    
    mask_pos= list(range(nan_pos-avoid_range,nan_pos+avoid_range+1))

    sorted_coords= [x - nan_pos for x in avail_coords]
    sorted_coords= {avail_coords[z]:sorted_coords[z] for z in range(len(avail_coords))}
    sorted_coords= sorted(sorted_coords,key= sorted_coords.get,reverse= True)
    print('hi')
    stp_idx= 0

    for stp in sorted_coords:
        
        #stp_idx= np.random.randint(0,len(avail_coords),1)[0]
        #stp= avail_coords[stp_idx]

        nwind= wind_extract_func(genotype, idx= stp, wind_sizes= wind_sizes,mask_pos= mask_pos)
        
        code_check= keep_tools[0](nwind,**keep_tools[1])
        code_check= np.array(code_check,dtype= int)
        
        if not code_check[nan_acc]:
            continue
        
        code_check=[code_check[x] for x in other_obs]
        nan_array.append(code_check)
        trail.append(stp)
        
        
    pres_dat= np.array(nan_array)
    feats= pca.fit_transform(pres_dat)
    print(pres_dat.shape)
    bandwidth = estimate_bandwidth(feats, quantile=0.2)
    
    if bandwidth == 0:
        labels1= [0]*feats.shape[0]
    else:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True, min_bin_freq=25)
        ms.fit(feats)
        labels1 = ms.labels_
    
    label_select = {y:[x for x in range(len(labels1)) if labels1[x] == y] for y in sorted(list(set(labels1)))}
    label_sizes= {z:len(g) for z,g in label_select.items()}
    lab_max= sorted(label_sizes,key= label_sizes.get,reverse= True)[0]
    ##
    ##
    if label_sizes[lab_max] >= Nrep:
        
        lab_chose= label_select[lab_max]
        pres_dat= pres_dat[lab_chose]
        
        pres_chose= np.sum(pres_dat,axis= 1)
        #
        pres_range= range(min(pres_chose),max(pres_chose)+1)
        pres_cdf= [sum(pres_chose >= x) for x in pres_range]                
        
        for idx in range(len(pres_cdf))[::-1]:
            if pres_range[idx] > ind_min:
                if pres_cdf[idx] >= Nrep:
                    pres_select= pres_chose >= pres_range[idx]
                    #
                    select_same= pres_dat[pres_select]
                    select_same= np.sum(select_same,axis= 0)
                    #
                    select_same= select_same == max(select_same)
                    select_same= np.array(other_obs)[select_same]
                    #
                    pres_select= np.array(lab_chose)[pres_select]
                    keep_select= np.random.choice(pres_select,Nrep,replace= False)
                    
                    Seq_store= [trail[x] for x in  pres_select]
    


    return select_same, Seq_store
    

