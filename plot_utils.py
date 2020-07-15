import plotly.graph_objects as go
import pandas as pd

def plot_map_comarca_points(data,cat,col,txt,max_size,max_value,x_low,x_up,y_low,y_up,title_name):
    fig = create_figure()
    fig = add_trace_plot(fig,data,col,txt,max_size,max_value,m_color='#63022d',series_name='Resposta covid')
    fig = plot_layout(fig,cat,x_low,x_up,y_low,y_up,title_name)
    return(fig)

def create_figure():
    fig = go.Figure()
    return(fig)

def add_trace_plot(fig,data,col,txt,max_size,max_value,m_color,series_name):
      # Add trace
    fig.add_trace(
        go.Scatter(x=data['longitude'],
                   y=data['latitude'],
                   mode='markers',
                   hoverinfo='text',
                  hovertext='Comarca: '+data['comarca_origin'] +\
                   '<br>' + 'Productors '+txt+': '+data[col].astype(int).astype(str),
                  marker=dict(size=data[col]*(max_size/max_value),
                             color=m_color),
                  name=series_name)
    )
    return(fig)

def add_trace_text_plot(fig,data,col,txt,max_size,max_value,m_color):
    if 'pctge' in col:
        text_col = data[col].astype(int).astype(str)+'%' 
    else:
        text_col = data[col].astype(int).astype(str)
    fig.add_trace(
        go.Scatter(x=data['longitude'],
                   y=data['latitude'],
                   text='<b>'+text_col + '</b>',
                   mode='markers+text',
                   hoverinfo='text',
                  hovertext='Comarca: '+data['comarca_origin'] +\
                   '<br>' + 'Productors '+txt+': '+data['total'].astype(int).astype(str),
                  marker=dict(size=data[col]*(max_size/max_value),
                             color='#63022d'),
                   textposition="top center"
                  )
    )
    return(fig)

def plot_layout(fig,cat,x_low,x_up,y_low,y_up,title_name):
      # Add images
    fig.add_layout_image(
            dict(
                source=cat,
                xref="x",
                yref="y",
                x=x_low,
                y=y_up,
                sizex=x_up-x_low,
                sizey=y_up-y_low,
                sizing="stretch",
                opacity=0.5,
                layer="below")
    )

    # Set templates
    fig.update_layout(template="plotly_white",
                      title=title_name,
                     yaxis=dict(range=[y_low,y_up],
                               showgrid=False,
                               showticklabels=False),
                     xaxis=dict(range=[x_low,x_up],
                               showgrid=False,
                               showticklabels=False),
                     width=800,
                     height=800)
    return(fig)


    

def plotly_hist(data, col, txt):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data[col],
        xbins=dict( # bins used for histogram
            start=data[col].min(),
            end=data[col].max(),
            size=1
        ),
        marker_color='#63022d',
        opacity=0.75
    ))

    fig.update_layout(template="plotly_white",
                      title=txt,
                     xaxis_title_text='Value',
                     yaxis_title_text='Count')

    fig.show()

def bar_perc_separate_datasets(data,col,txt):
    rep_txt = txt.replace('Mitja','Percentatge')
    ab,pag = separate_ab_from_pag_data(data)

    ab_gb = ab.groupby(col)['MARCA'].count()/ab.shape[0]*100
    ab_gb = ab_gb.round(2)
    
    pag_gb = pag.groupby(col)['MARCA'].count()/pag.shape[0]*100
    pag_gb = pag_gb.round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ab_gb.index,
        y=ab_gb,
        marker_color='#63022d',
        opacity=0.75,
        name='Abastiment'
    ))

    fig.add_trace(go.Bar(
        x=pag_gb.index,
        y=pag_gb,
        marker_color='#f8cecc',
        opacity=0.75,
        name='Pagesos'
    ))

    fig.update_layout(template="plotly_white",
                      title=rep_txt,
                     xaxis_title_text=txt,
                     yaxis_title_text='Percentatge productors (%)')

    return(fig)


def separate_ab_from_pag_data(data):
    ab  = data.loc[(data.dataset=='abastiment')]
    pag = data.loc[(data.dataset=='pagesos')]
    return(ab,pag)
    
def pagament_prep(data,pagament):
    ab,pag = separate_ab_from_pag_data(data)
    ab.rename(columns=pagament,inplace=True)
    pag.rename(columns=pagament,inplace=True)

    ab_gb = pd.DataFrame(ab[list(pagament.values())].sum(),columns=['sum'])
    ab_gb['pctge'] = ab_gb['sum']/ab.shape[0]*100
    #ab_gb = ab_gb.sort_values(by='pctgeascending=False)
    ab_gb = ab_gb.round(2)
    
    pag_gb = pd.DataFrame(pag[list(pagament.values())].sum(),columns=['sum'])
    pag_gb['pctge'] = pag_gb['sum']/pag.shape[0]*100
    pag_gb = pag_gb.sort_values(by='pctge',ascending=False)
    pag_gb = pag_gb.round(2)
    return(ab_gb,pag_gb)

def get_n_data_per_dataset(data,n_comarca):
    n_abastiment = data[data.dataset=='abastiment'].groupby(['dataset','comarca_origin'],
                             as_index=False)['MARCA'].count().rename(columns={
                                                                    'MARCA':'n_abastiment'})
    n_pagesos = data[data.dataset=='pagesos'].groupby(['dataset','comarca_origin'],
                             as_index=False)['MARCA'].count().rename(columns={
                                                                    'MARCA':'n_pagesos'})
    n_dataset = n_comarca.merge(n_abastiment.drop('dataset',axis=1), 
                                on='comarca_origin',
                                how='right').merge(n_pagesos.drop('dataset',axis=1),
                                                   on='comarca_origin',
                                                   how='outer')
    return(n_dataset)

def dataset_to_plot(data,vdp,com_coord,n_columns,multiple_origins=False):
    '''Counts the values per comarca for the whole dataset and per dataset tipe, 
    computes the mean for those columns that are results of sums, sums the values
    of the other columns and returns the resulting dataset with data per comarca'''
    n_new = data.groupby('comarca_origin',
                         as_index=False)['MARCA'].count().rename(columns={
                                                                'MARCA':'total'})
    # number of producers existing before
    n_old = vdp.groupby('comarca_origin',as_index=False)['MARCA'].count().rename(columns={
                                                                'MARCA':'n_before'})
    n_comarca = n_new.merge(n_old)
    n_comarca['pctge_new'] = n_comarca['total']/n_comarca['n_before']*100
    n_comarca['pctge_new'] = n_comarca['pctge_new'].astype(int)

    # if the input dataset contains data from abastiment + pagesos, we will 
    # compute the number per dataset also 
    if multiple_origins == True:
        n_comarca = get_n_data_per_dataset(data,n_comarca)

    mean_dataset = data[['comarca_origin'] + n_columns].groupby('comarca_origin',
                                                     as_index=False).mean().round(2)
    sum_dataset = data.drop(n_columns,axis=1).groupby('comarca_origin',
                                                  as_index=False).sum().merge(com_coord,
                                                          left_on='comarca_origin',
                                                         right_on='comarca',
                                                                             how='left')
    to_plot = n_comarca.merge(sum_dataset,
                          on='comarca_origin',
                          how='outer').merge(mean_dataset,
                                            on='comarca_origin',
                                            how='outer').fillna(0)
    return(to_plot)


def bar_payment_type(pag_gb,ab_gb):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pag_gb.index,
        y=pag_gb['pctge'],
        hoverinfo='text',
        hovertext='Percentatge amb el tipus de pagament: '+pag_gb['pctge'].astype(str) +'%'+\
                   '<br>' + 'Número de productors corresponent : '+pag_gb['sum'].astype(int).astype(str),
        marker_color='#f8cecc',
        opacity=0.75,
        name='Pagesos'
    ))
    
    fig.add_trace(go.Bar(
        x=ab_gb.index,
        y=ab_gb['pctge'],
        hoverinfo='text',
        hovertext='Percentatge amb el tipus de pagament: '+ab_gb['pctge'].astype(str) +'%'+\
                   '<br>' + 'Número de productors corresponent : '+ab_gb['sum'].astype(int).astype(str),
        marker_color='#63022d',
        opacity=0.75,
        name='Abastiment'
    ))

    

    fig.update_layout(template="plotly_white",
                      title='Mètodes de pagament',
                     xaxis_title_text='Tipus de pagament',
                     yaxis_title_text='Percentatge productors (%)')
    return(fig)

def hist_separate_datasets(data, col, txt):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data.loc[(data.dataset=='abastiment'), col],
        xbins=dict( # bins used for histogram
            start=data[col].min(),
            end=data[col].max(),
            size=1
        ),
        marker_color='#63022d',
        opacity=0.75,
        name='Abastiment'
    ))

    fig.add_trace(go.Histogram(
        x=data.loc[(data.dataset=='pagesos'), col],
        xbins=dict( # bins used for histogram
            start=data[col].min(),
            end=data[col].max(),
            size=1
        ),
        marker_color='#f8cecc',
        opacity=0.75,
        name='Pagesos'
    ))

    fig.update_layout(template="plotly_white",
                      title=txt,
                     xaxis_title_text='Value',
                     yaxis_title_text='Count')

    fig.show()