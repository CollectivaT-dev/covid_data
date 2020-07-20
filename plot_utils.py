import plotly.graph_objects as go
from ipysankeywidget import SankeyWidget
from floweaver import weave, ProcessGroup, Bundle, Partition, SankeyDefinition, QuantitativeScale
import pandas as pd
import numpy as np
import preprocess as prep

def plot_map_comarca_points(data,cat,col,txt,max_value,title_name):
    fig = create_figure()
    fig = add_trace_plot(fig,data,col,txt,max_value,m_color='#63022d',series_name='Resposta covid')
    fig = plot_layout(fig,cat,title_name)
    return(fig)

def create_figure():
    fig = go.Figure()
    return(fig)

def add_trace_plot(fig,data,col,txt,max_value,m_color,series_name):
    max_size = 40
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

def add_trace_text_plot(fig,data,col,txt,max_value,m_color):
    max_size = 40
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

def plot_layout(fig,cat,title_name):
    x_low,x_up = 0.18,3.3
    y_low,y_up = 40.5,42.9
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
        marker_color='#f7b49d',
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

    ab_gb = pd.DataFrame(ab[list(pagament.values())].replace('', np.nan).sum(),columns=['sum'])
    ab_gb['pctge'] = ab_gb['sum']/ab.shape[0]*100
    #ab_gb = ab_gb.sort_values(by='pctgeascending=False)
    ab_gb = ab_gb.round(2)
    
    pag_gb = pd.DataFrame(pag[list(pagament.values())].replace('', np.nan).sum(),columns=['sum'])
    pag_gb['pctge'] = pag_gb['sum']/pag.shape[0]*100
    pag_gb = pag_gb.sort_values(by='pctge',ascending=False)
    pag_gb = pag_gb.round(2)
    return(ab_gb,pag_gb)



def dataset_to_plot(data,vdp,com_coord,multiple_origins=False):
    '''Counts the values per comarca for the whole dataset and per dataset tipe, 
    computes the mean for those columns that are results of sums, sums the values
    of the other columns and returns the resulting dataset with data per comarca'''
    n_columns = ['n_main_prod','n_other_prod','n_tot_prod',
                'n_paym_methods','n_comarcas_delivery']
    n_new = data.groupby('comarca_origin',
                         as_index=False)['MARCA'].count().rename(columns={
                                                                'MARCA':'total'})
    # number of producers existing before
    n_old = vdp.groupby('comarca_origin',as_index=False)['MARCA'].count().rename(columns={
                                                                'MARCA':'n_before'})
    n_comarca = n_new.merge(n_old)
    n_comarca['pctge_new'] = n_comarca['total']/n_comarca['n_before']*100
    n_comarca['pctge_new'] = n_comarca['pctge_new'].astype(int)

    data = data.replace('', np.nan)
    # if the input dataset contains data from abastiment + pagesos, we will 
    # compute the number per dataset also 
    if multiple_origins == True:
        n_comarca = get_n_data_per_dataset(data,n_comarca)

    n_columns = [col for col in n_columns if col in data.columns]
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


def bar_payment_type(pag_gb,ab_gb):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pag_gb.index,
        y=pag_gb['pctge'],
        hoverinfo='text',
        hovertext='Percentatge amb el tipus de pagament: '+pag_gb['pctge'].astype(str) +'%'+\
                   '<br>' + 'Número de productors corresponent : '+pag_gb['sum'].astype(int).astype(str),
        marker_color='#f7b49d',
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


def plot_sankey_sector(data, com_coord, save=False):
    sector_list = prep.read_yaml('conf','subsets_criteria')
    paths = prep.read_yaml('conf','paths')
    for sector in sector_list.keys():
        print('-----', sector,'subset -----')
        dic = sector_list[sector]
        
        data_sel = filter_sector_subset(data, dic['on_fields'], dic['off_fields'])

        #Selecting only the desired subset of producers
        if data_sel.shape[0] == 0:
            print('There are no producers in the requested subset: ', subset)
            continue
        else:
            #print('Dimension of the subset: ', data_sel.shape)
            flows = create_df_for_sankey(data_sel)
            ###https://github.com/psychemedia/parlihacks/blob/master/notebooks/MigrantFlow.ipynb)

            sdd = plot_sankey(flows, com_coord)
            ## New Sankey!
            size = dict(width=870, height=1000)
            weave(sdd, flows).to_widget(**size) 
            
            display(weave(sdd, flows, link_color=QuantitativeScale('value'), \
                measures='value').to_widget(**size))
            if save:
                ## Saving the plot as svg
                name = paths['output'] + "sankeydiag_"+sector+".svg"
                weave(sdd, flows, link_color=QuantitativeScale('value'), \
                measures='value').to_widget(**size).auto_save_svg(name)
                #print('File saved in: ', name)

def filter_sector_subset(data, on_fields, off_fields):
    '''Filter the input data so only producers from a specific 
    sector are kept. The filter will be based on what fields the
    sector should have data on and which souldn't'''

    if on_fields is None and off_fields is None:
        data['is_subset'] = 1
        
    elif on_fields is None and off_fields is not None:
        data['is_subset'] = (np.where(data[off_fields].eq(0).all(axis=1), 1, 0))
        
    elif on_fields is not None and off_fields is None:
        data['is_subset'] = (np.where(data[on_fields].eq(1).all(axis=1), 1, 0))
        
    elif on_fields is not None and off_fields is not None:
        data['flags_on']  = (np.where(data[on_fields].replace('', np.nan).ge(1).all(axis=1), 1, 0))
        data['flags_off'] = (np.where(data[off_fields].eq(0).all(axis=1), 1, 0))
        data['is_subset'] = np.where(data[['flags_on', 'flags_off']].sum(axis=1).eq(2),1,0)
        data.drop(['flags_on','flags_off'], axis=1, inplace=True) 
        
    data = data[data.is_subset == 1][['DONDE','comarca_origin']]
    data = data.query('not DONDE.str.contains("NOTFOUND") and not comarca_origin.str.contains("NOTFOUND")')
    
    return(data)

def create_df_for_sankey(data):
    '''Connections between comarcas. Creating the dataframe needed for 
    sankey diagram (i.e. the list of all the edges between two comarcas),
    it will have the following columns: source, target, value.'''
    
    #Extracting all the target comarcas from the field 'DONDE'
    all_cat_ind = data[data.DONDE.str.count(', ') >= 40].index
    data.loc[all_cat_ind,'DONDE'] = 'Catalunya'
    data['value'] = 1
    data.rename(columns={'comarca_origin':'source','DONDE':'target'},inplace=True)
    data['target'] = data['target'].str.split(', ')
    data = data.explode('target')
    data = data.replace('', np.nan)
       
    #Creation of the final df by grouping by (source, target) couples
    
    #Getting the normalization factor (i.e. the total number of connections per comarca of origin)  
    df_norm = data.groupby(['source'])['value'] \
                             .sum() \
                             .reset_index(name='norm_factor') 

    #Grouping by the connections with same source-target: 
    df_edges = data.groupby(['source', 'target'])['value'] \
                             .sum() \
                             .reset_index(name='value') \
                             .sort_values(['value'], ascending=False) \

    #Adding the normalized factor to the edges df:
    df_edges = pd.merge(df_edges, df_norm, how='inner', left_on='source', right_on='source')
    df_edges['norm_value'] = df_edges['value'].astype(float)/df_edges['norm_factor'].astype(float)*100

    return(df_edges[['source', 'target', 'value']])

def plot_sankey(flows, com_coord):
    com_coord = com_coord.append({'comarca':'Catalunya','latitude':0,'longitude':0},ignore_index=True)

    SankeyWidget(links=flows.to_dict('records'))        

    nodes = {
        'Comarcas_productoras': ProcessGroup(list(com_coord.comarca.unique())),
        'Comarcas_entrega'    : ProcessGroup(list(com_coord.comarca.unique())),
    }
    # productoras on the left, entrega on the right
    ordering = sorted([[key] for key,_ in nodes.items()],reverse=True) 
    bundles = [
        Bundle(sorted(list(nodes),reverse=True)[0],
               sorted(list(nodes),reverse=True)[1]),
    ]
    comarcas = Partition.Simple('process',list(com_coord.comarca.unique()))
    
    # Update the ProcessGroup nodes to use the partitions
    nodes['Comarcas_productoras'].partition = comarcas
    nodes['Comarcas_entrega'].partition = comarcas

    sdd = SankeyDefinition(nodes, bundles, ordering)

    return(sdd)