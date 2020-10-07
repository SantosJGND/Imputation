import plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import plot, iplot

import numpy as np

def plot_extracted(featl, label_select, tf_acc, labels= [],plot_out= True):

    if not labels:
        labels= ['i{}'.format(x) for x in range(featl.shape[0])]

    figwl= [go.Scatter(
        x= featl[label_select[i],0],
        y= featl[label_select[i],1],
        text= [labels[x] for x in label_select[i]],
        mode= 'markers',
        name= str(i),
        marker= dict(
            size= 10
        )
    ) for i in label_select.keys()]


    figwl.append(go.Scatter(
        mode='markers',
        #x= [pca_w.transform(local_l[tf_acc].reshape(1,-1))[0][0]],
        #y= [pca_w.transform(local_l[tf_acc].reshape(1,-1))[0][1]],
        x=[featl[tf_acc,0]],
        y=[featl[tf_acc,1]],
        marker=dict(
            color='rgba(135, 206, 250, 0)',
            size=25,
            opacity= 1,
            line=dict(
                color='blue',
                width=5
            )
        ),
        showlegend=False
    ))

    layout= go.Layout()

    Figure_wl= go.Figure(data= figwl, layout= layout)

    if plot_out:
        iplot(Figure_wl)
    
    return figwl



def plot_compare(figwl, background, like_diet, tf_proj):
    
    title= 'coords'
    fig_subplots = tools.make_subplots(rows=1, cols=2,subplot_titles=tuple([title]*2))

    for trace in figwl:
        fig_subplots.append_trace(trace, 1, 1)

    correct= max(like_diet)

    if correct==0:
        correct= 1

    if background.shape[1] > 2:
        opac = like_diet / correct 
        opac= opac
    else:
        opac= .8

    trace= go.Scatter(
        x= background[:,0],
        y= background[:,1],
        #z= grid_likes,
        mode= 'markers',
        marker= {
            'color': like_diet,
            'colorbar': go.scatter.marker.ColorBar(
                            title= 'likelihood',
                            yanchor="top", y=0.3,
                            lenmode="pixels", len=200,
                        ),
            'colorscale':'Viridis',
            'line': {'width': 0},
            'size': 25,
            'symbol': 'circle',
          "opacity": opac #like_diet / correct
          }
    )

    like_max= np.argmax(like_diet)

    target_found= go.Scatter(
        mode='markers',
        #x= [pca_w.transform(local_l[tf_acc].reshape(1,-1))[0][0]],
        #y= [pca_w.transform(local_l[tf_acc].reshape(1,-1))[0][1]],
        #x= [background[like_max][0]],
        #y= [background[like_max][1]],
        x= [tf_proj[0]],
        y= [tf_proj[1]],
        name= 'max_predict',
        marker=dict(
            color='rgba(135, 206, 250, 0)',
            size= 35,
            opacity= 1,
            line=dict(
                color='red',
                width=5
            )
        ),
        showlegend=False
    )

    fig_subplots.append_trace(target_found, 1,1)
    fig_subplots.append_trace(trace, 1,2)

    iplot(fig_subplots)

    return trace



###
###



def plot_stats(sts_grid, titles, Ncols= 2, wind_sizes= 20):

    fig_subplots = tools.make_subplots(rows= int(len(sts_grid) / float(Ncols)) + (len(sts_grid) % Ncols > 0), cols=Ncols,
                             subplot_titles=tuple(titles))

    #####
    for gp in range(len(titles)):
        facvec= sts_grid[gp]

        pos1= int(float(gp) / Ncols) + 1
        pos2= gp - (pos1-1)*Ncols + 1

        trace= go.Box(
            y= facvec
        )
        
        fig_subplots.append_trace(trace, pos1, pos2)

    layout = go.Layout(
        title= 'benchmark stats',
    )
    
    fig_subplots['layout'].update(height= 900, title= 'benchmark stats, window sizes= {}'.format(wind_sizes))
    
    fig= go.Figure(data=fig_subplots, layout=layout)
    iplot(fig_subplots)


###
###