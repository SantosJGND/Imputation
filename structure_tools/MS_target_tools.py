
import scipy
import itertools as it
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import pairwise_distances
from plotly.offline import iplot


def avoid_idx(data,char_avoid= [9,1],ind_thresh= 0.01):
    """
    get index of samples with proportion of features in char_avoid > ind_thresh.
    """
    nstat= np.zeros(data.shape)
    for ix in char_avoid:
        nm= data == ix
        nstat+= nm

    nstat= np.array(nstat,dtype= int)

    nstat= np.sum(nstat,axis= 1)
    nstat= nstat / float(data.shape[1])

    nstat= nstat >= ind_thresh

    nstat= np.array(nstat,dtype= int)
    #
    return nstat


def MS_get_norm(Sequences,refs_lib,ncomps= 4,clsize= 15,Bandwidth_split= 20,
               pca_qtl= 0.2, char_avoid= [9,1],ind_thresh= 0.01):
    '''
    Perform PCA + Mean Shift across windows. Extract Meanshift p-value vectors. Perform amova (optional).
    '''

    nstat= avoid_idx(Sequences,char_avoid= char_avoid,ind_thresh= ind_thresh)
    avoid= [x for x in range(len(nstat)) if nstat[x] == 1]
    keep= [x for x in range(len(nstat)) if nstat[x] == 0]

    pca = PCA(n_components=ncomps, whiten=False,svd_solver='randomized').fit(Sequences[keep])
    data = pca.transform(Sequences)

    params = {'bandwidth': np.linspace(np.min(data), np.max(data),Bandwidth_split)}
    grid = GridSearchCV(KernelDensity(algorithm = "ball_tree",breadth_first = False), params,verbose=0,cv= 3, iid= False)

    ######################################
    ####### TEST global Likelihood #######
    ######################################
    Focus_labels = [z for z in it.chain(*refs_lib.values()) if z not in avoid]

    #### Mean Shift approach
    ## from sklearn.cluster import MeanShift, estimate_bandwidth

    bandwidth = estimate_bandwidth(data, quantile= pca_qtl, n_samples=len(Focus_labels))
    if bandwidth <= 1e-3:
        bandwidth = 0.1

    ms = MeanShift(bandwidth=bandwidth, cluster_all=False, min_bin_freq=clsize)
    ms.fit(data[Focus_labels,:])
    labels = ms.labels_

    Tree = {x:[Focus_labels[y] for y in range(len(labels)) if labels[y] == x] for x in [g for g in list(set(labels)) if g != -1]}
    Keep= [x for x in Tree.keys() if len(Tree[x]) > clsize]

    Tree= {x:Tree[x] for x in Keep}
    Ngps= len(Tree)

    ### Extract MScluster likelihood by sample

    dist_store= {}

    for hill in Tree.keys():
        
        grid.fit(data[Tree[hill],:])

        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_

        # normalize kde derived log-likelihoods, derive sample p-values
        P_dist = kde.score_samples(data[Tree[hill],:])
        Dist = kde.score_samples(data)
        P_dist= np.nan_to_num(P_dist)
        Dist= np.nan_to_num(Dist)
        
        if np.std(P_dist) == 0:
            Dist= np.array([int(Dist[x] in P_dist) for x in range(len(Dist))])
        else:
            Dist = scipy.stats.norm(np.mean(P_dist),np.std(P_dist)).cdf(Dist)
            Dist= np.nan_to_num(Dist)
            Dist[avoid]= 0

            dist_store[hill]= Dist
    
    return Tree, dist_store,data



def kde_gen_dict(data,label_dict,
                    Bandwidth_split= 20):
    '''
    create dictionary of group kde generators in data space.
    '''
    
    params = {'bandwidth': np.linspace(np.min(data), np.max(data),Bandwidth_split)}
    grid = GridSearchCV(KernelDensity(algorithm = "ball_tree",breadth_first = False), params,verbose=0,cv= 3, iid= False)

    ref_gens= {}
    ref_stats= {}

    for hill in label_dict.keys():

        grid.fit(data[label_dict[hill],:])
        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_
        ref_gens[hill]= kde
        
        kd_scores= kde.score_samples(data[label_dict[hill],:])
        kd_stats= [np.mean(kd_scores),np.std(kd_scores)]
        ref_stats[hill]= kd_stats
    
    return ref_gens, ref_stats

from scipy.stats import norm



def gen_class(samples,ref_generators,gen_stats= {},lb= 1e-3,out_code= -1):
    '''
    use kde generators in dictionary to score and classify samples.
    '''
    ref_keys= list(ref_generators.keys())
    score_dict= {z: g.score_samples(samples) for z,g in ref_generators.items()}
    if gen_stats:
        
        score_dict= {z: norm.cdf(g,loc= gen_stats[z][0],scale= gen_stats[z][1]) for z,g in score_dict.items()}
    #print([x.shape for x in score_dict.values()])
    score_array= [score_dict[z] for z in ref_keys]
    score_array= np.array(score_array)
    #score_array= np.exp(score_array)
    
    maxs= np.max(score_array,axis= 0)
    #print(maxs)
    maxs= maxs < lb
    
    score_sum= np.sum(score_array,axis= 0)
    score_sum[score_sum == 0]= 1
    score_array= score_array / score_sum
    
    maxl= np.argmax(score_array,axis= 0)

    maxl= np.array(ref_keys)[maxl]
    maxl[maxs]= out_code
    
    return maxl


def clustClass(ms_local,pca_obj,ref_gens,gen_stats= {},out_code= -1, 
               return_mean= True,lb= 1e-2):
    '''
    ms_local= distances by cluster.
    '''
    
    mskeys= list(ms_local.keys())

    ## 
    dist_array= [ms_local[g] for g in mskeys]
    dist_array= np.array(dist_array)
    qtl_dist= pca_obj.transform(dist_array)
    #print(qtl_dist.shape)
    ## Classify kde profiles. 
    cluster_class= gen_class(qtl_dist,ref_gens,gen_stats= gen_stats,lb= lb, 
                             out_code= out_code)
    
    
    cluster_found= {z: [x for x in range(len(cluster_class)) if cluster_class[x] == z] for z in list(set(cluster_class)) if z != out_code}

    for v,g in cluster_found.items():
        dist_foud= qtl_dist[g]
        if dist_foud.shape[0] > 1:
            dist_foud= np.mean(dist_foud,axis= 1)

        g= dist_foud    
    
    return cluster_found



def D1_kdegen(dists_dict,kernel='gaussian', bandwidth=0.05):
    '''
    '''
    gen_dict= {}
    for gp,data in dists_dict.items():
        
        if not data:
            gen_dict[gp]= data
            continue
        data= np.array(data).reshape(-1,1)
        
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
        gen_dict[gp]= kde

    return gen_dict



from plotly import subplots
import plotly
import plotly.graph_objs as go

def plot_distances(dists_dict,gp,range_dists,height= 500,width= 900):
    Ncols= 1
    
    keys_get= sorted([v for v,g in dists_dict[gp].items() if len(g)])
    titles= ['cl: {}'.format(g) for g in keys_get]
    print(titles)
    
    dist_gens= {}
    
    if titles:
        fig_subplots = subplots.make_subplots(rows= int(len(titles) / float(Ncols)) + (len(titles) % Ncols > 0), cols=Ncols,
                                 subplot_titles=tuple(titles))

        for idx in range(len(titles)):
            print(idx)
            ref= keys_get[idx]
            pos1= int(float(idx) / Ncols) + 1
            pos2= idx - (pos1-1)*Ncols + 1

            title= titles[idx]

            data= dists_dict[gp][ref]
            data= np.array(data).reshape(-1,1)
            kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(data)
            dist_gens[ref]= kde
            
            scor_dist= kde.score_samples(range_dists)
            scor_dist= np.exp(scor_dist)
            trace1= go.Scatter(
                y= scor_dist,
                x= range_dists.T[0],
                mode= 'markers',
                name= titles[idx]
            )

            fig_subplots.append_trace(trace1, pos1, pos2)

            fig_subplots['layout']['yaxis' + str(idx + 1)].update(title= 'L')
            fig_subplots['layout']['yaxis' + str(idx + 1)].update(range= [0,max(scor_dist) + max(scor_dist)/10])
            fig_subplots['layout']['xaxis' + str(idx + 1)].update(title= 'pca dist')

        layout = go.Layout(
            title= title,
        )

        fig= go.Figure(data=fig_subplots, layout=layout)

        fig['layout'].update(height= height,width= width)

        iplot(fig)
        
        return dist_gens



def target_MSP(SequenceStore,preProc_Clover, comp_label_keep, refs_lib, Whose,
               ncomps= 4, clsize= 15, Bandwidth_split= 20, out_code= -1,
               metric= 'euclidean', cl_samp= 50, pca_qtl= .2,
               char_avoid= [9,1],ind_thresh= 0.01):
    '''
    get dictionary of feature windows.
    Get reference MS profile kde and stats preProc_Clover and comp_label_keep.
    preProc_Clover = ms profile array;
    comp_label_keep = index dictionary of MS profile groups to use as reference. 
    For every window:
        - Identify and classify clusters by window.
        - Calculate distances between identified clusters (use kde to generate cl_samp samples)
    
    return dictionary of distances between target clusters. 
    '''
    clov_pca= PCA(n_components=ncomps, whiten=False,svd_solver='randomized').fit(preProc_Clover)
    data_clov= clov_pca.transform(preProc_Clover)


    ref_gens, ref_stats= kde_gen_dict(data_clov,comp_label_keep,
                                        Bandwidth_split= Bandwidth_split)
    dists_dict= {z:{y:[] for y in ref_gens.keys()} for z in ref_gens.keys()}
    
    for c in SequenceStore.keys():

        ### PCA.
        Sequences= [SequenceStore[c][x] for x in Whose]
        Sequences= np.array(Sequences) 
        Sequences= np.nan_to_num(Sequences)

        lclust_samp, lclust_gens= clust_samp(Sequences, refs_lib, clov_pca, ref_gens, ref_stats,
              ncomps= ncomps,clsize= clsize,Bandwidth_split= Bandwidth_split,cl_samp= cl_samp,
                pca_qtl= pca_qtl, char_avoid= char_avoid,ind_thresh= ind_thresh)

        if not lclust_samp:
            continue

        lclust_means= {z: np.mean(g,axis= 0) for z,g in lclust_samp.items()}
        cluster_keys= list(lclust_samp.keys())

        ####
        hills= [lclust_means[z] for z in cluster_keys]
        hills= np.array(hills)
        
        hill_dists= pairwise_distances(hills,metric= metric)

        for idx in range(len(cluster_keys)):
            for idx1 in range(len(cluster_keys)):
                if idx != idx1:
                    cd1= cluster_keys[idx]
                    cd2= cluster_keys[idx1]
                    dists_dict[cd1][cd2].append(hill_dists[idx,idx1])
    
    return dists_dict




def clust_samp(local_l, refs_lib, clov_pca, ref_gens, ref_stats,
              ncomps= 4,clsize= 15,Bandwidth_split= 20, cl_samp= 50,
                pca_qtl= .2, out_code= -1, return_feats= False, return_acc= False,
                char_avoid= [9,1],ind_thresh= 0.01):
    '''
    classify local clusters, sample uing kde gens. 
    '''
    clust_acc, ms_local, feat_seq= MS_get_norm(local_l,refs_lib,ncomps= ncomps,clsize= clsize,Bandwidth_split= Bandwidth_split,
           pca_qtl= pca_qtl, char_avoid= char_avoid, ind_thresh= ind_thresh)

    mskeys= list(ms_local.keys())

    ####
    cluster_found= clustClass(ms_local,clov_pca,ref_gens,gen_stats= ref_stats,out_code= out_code)

    cluster_found= {g[0]:z for z,g in cluster_found.items()}

    clust_acc= {z: g for z,g in clust_acc.items() if z in cluster_found.keys()}
    clust_acc= {cluster_found[z]:g for z,g in clust_acc.items()}

    ####
    cluster_keys= list(clust_acc.keys())

    lclust_gens, lclust_stats= kde_gen_dict(feat_seq,clust_acc)

    lclust_samp= {z:g.sample(cl_samp) for z,g in lclust_gens.items()}
    lclust_means= {z: np.median(g,axis= 0) for z,g in lclust_samp.items()}
    
    if return_acc:
        return lclust_samp, lclust_gens, clust_acc
    
    if return_feats:
        return lclust_samp, lclust_gens, feat_seq
    
    else:
        return lclust_samp, lclust_gens



def comb_score(background,lclust_samp= {},dists_gens= {},
    select_missing= 0,dimN= 2, metric= "euclidean"):
    '''
    score background array of coordinates against a set of reference coordinates.
    likelihood is calculated on score_sample equipped objects on distance values (1D).
    '''
    dist_refs= {}
    dist_refs= {
        z: pairwise_distances(background,g[:,:dimN],metric= metric) for z,g in lclust_samp.items()
    }
    
    dist_refMeans= {z: np.mean(g,axis= 1) for z,g in dist_refs.items()}
    
    select_gens= {}

    for gp in lclust_samp.keys():
        g= dists_gens[gp]
        
        if g[select_missing]:
            select_gens[gp]= g[select_missing]
    
    ##
    bg_score= {z: g.score_samples(dist_refMeans[z].reshape(-1,1)) for z,g in select_gens.items()}
    
    bg_scores= np.array(list(bg_score.values()))
    bg_scores= np.exp(bg_scores)
    
    ##
    return bg_scores



