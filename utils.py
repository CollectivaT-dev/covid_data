import re
import unicodedata
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from stop_words import get_stop_words
from  geopy.geocoders import Nominatim
import time
import pandas as pd
import numpy as np

def get_all_stopwords():
    #load a set of stop words
    stopwords = get_stop_words('catalan')
    #add new stopwords
    newStopWords = ['que','des', 'al', 'del', 'ho', 'd', 'l','per','tambe', 'fins',
                   'a', 'cap', 'hi', 'ni', 'no']
    stopwords.extend(newStopWords)
    return(stopwords)

def run_preprocess_on_cols(data,cols,stopwords):
    for col in cols:
        data[col+'_prep'] = data[col].apply(lambda x: pre_process(x,stopwords,sw=True))

def pre_process(text, stopwords=[], sw=False):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    ## remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    # remove only digits
    #text=re.sub("(\\d)+"," ",text)

    # remove accents
    text=re.sub("[\u0300-\u036f]", "", unicodedata.normalize('NFD', text))
    text=text.strip()
    #text=unidecode(text) ##if I do it I need to do it also in the stopwords
    
    if sw:
        stopwords = [re.sub("[\u0300-\u036f]", "", 
               unicodedata.normalize('NFD', x)) for x in stopwords]
        text = re.sub(r'\s+',' ',re.sub(
            r'\b'+r'\b|\b'.join(stopwords)+r'\b','',text)).strip()
    
    return text

def check_comarca_spelling(df_text,comarca_df,stopwords):
    '''Homogenize the comarca spelling data.
    Compare the comarca spelling from the dataframe with the one
    from the locations data (both preprocessed). If the dataframe one 
    is found, the value from the locations data will be return.
    Useful to recognize comarcas without the correct accentuation
    missing determinants...'''
    comarca_df = comarca_df.unique()
    coms_prep  = [pre_process(x,stopwords,sw=True) for x in comarca_df]
    loc_prep   = pre_process(df_text, stopwords,sw=True)
    if loc_prep in coms_prep:
        ind = coms_prep.index(loc_prep)
        return(comarca_df[ind])
    else:
        print('Not found:',df_text)
        return('(NOTFOUND)')

def deliver_to(row, col_to_search, locations_df, look_for, loc_to_iterate, patt_to_search):
    '''obtain the delivery locations given free text, 
    a list of possible locations to look for
    and patterns that indicate delivery specifications
    '''
    txt = row[col_to_search+'_prep']
    locations_in_text = ''
    # we look for sentences that look like delivery specifications
    if re.search(patt_to_search,txt):
        relevant_txt = re.search(patt_to_search,txt).group().strip()
        for ind,loc in loc_to_iterate.iteritems():
            if re.search(r'\b'+loc+r'\b',relevant_txt):
                if not re.search(r'\b'+loc+r'\b',locations_in_text):
                    locations_in_text = locations_in_text + ','+locations_df[look_for][ind]
            
    elif col_to_search in ['comarca_new']:
        for ind,loc in loc_to_iterate.iteritems():
            if re.search(r'\b'+loc+r'\b',txt):
                if not re.search(r'\b'+loc+r'\b',locations_in_text):
                    locations_in_text = locations_in_text + ','+locations_df[look_for][ind]
                    
    return locations_in_text.strip(',')

def get_payment_methods(data,imp_cols):
    '''creates a new column containing a list of the different payment
    methods found in the free text fields'''
    data['PAGO'] = ''
    pago = {'efectiu':'efectiu', 
            'bizum':'bizum', 
            'transferencia previa':'transferencia', 
            'targeta':'targeta'}
    for c in [x+'_prep' for x in imp_cols]:
        for k,v in pago.items():
            if data[data[c].str.contains(r'\b'+v+r'\b')].shape[0]!=0:
                ind = data[data[c].str.contains(r'\b'+v+r'\b')].index
                data.loc[ind,'PAGO'] = data.loc[ind]['PAGO'] +','+ k
    data['PAGO'] = data['PAGO'].str.strip(',')
    return(data)

def get_text_locations(df, output_col,col_to_search, locations_df, look_for, delivery_patt):
    '''Run through the registers of the specified column and look for locations
    by calling deliver_to function'''
    loc_to_iterate = locations_df[(locations_df[look_for+'_prep'].notnull()) & 
                                     (locations_df[look_for+'_prep'] != '')
                                     ][look_for+'_prep'].drop_duplicates()
    patt_to_search = '(?:'+'|'.join(delivery_patt)+').*' 
    
    # if the output column already exists we want to concatenate the results, not replace them
    if output_col in df.columns:
        df[output_col] = df[output_col] + ',' + df.apply(lambda row: 
                                                         deliver_to(row, col_to_search, locations_df, look_for,
                                                                    loc_to_iterate, patt_to_search),axis=1)
        df[output_col] = df[output_col].str.strip(',')
    else:
        df[output_col] = df.apply(lambda row: deliver_to(row, col_to_search, locations_df, look_for,
                                                         loc_to_iterate, patt_to_search),axis=1)

def run_text_locations(data, locations_df, imp_cols, delivery_patt):
    for data_field in imp_cols+['comarca_origin']:
        for loc in ['Comarca','Capital','Provincia','Municipi']:
            # obtain the locations from the free text fields
            get_text_locations(data,loc.lower(),data_field,locations_df,loc,delivery_patt)

def create_donde_col(data,mun_to_com_dict):
    '''only for abastiments. create 'donde' column completing comarca_origin column with:
    - comarca data
    - replacing municipis with comarques for capital and municipi columns
    concatenating the resulting columns and keeping only the unique values'''
    #data.loc[data.comarca_origin.str.contains('NOTFOUND'), 'comarca_origin'] = data.COMARCA
    data[['capital','municipi']] = data[['capital','municipi']].replace(mun_to_com_dict,regex=True)
    data['DONDE']        = (data['capital']+','+data['municipi']+','+data['comarca_origin']
                                   ).str.strip(',').str.split(',')
    data.drop(['capital','municipi', 'comarca_origin_prep','comarca','provincia'],axis=1,inplace=True)
    data['DONDE'] = data['DONDE'].apply(lambda x: ', '.join(set(x)))
    # to have the same format as pagesos data
    data['DONDE'] = data['DONDE'].str.replace(r'\bCatalunya\b','Tota Catalunya')
    return(data)

def create_binary_var(data,dic,col):
    '''Create a column with binary values based on if the input column has the 
    text contained in the values of the input dictionary or not'''
    for key, val in dic.items():
        data[key]=0
        data.loc[data[col].str.contains(r'\b'+r'\b|\b'.join(val)+r'\b'),key] = 1

def create_sectors_col(df,dic):
    '''Create 5 columns with binary values representing if a producer belong to a given sector or not, 
    as defined in the input dictionary (at least one column of the list of columns related to the sector must be 1)'''
    for txt, sector in dic.items():
        #print('sector name: ', txt)
        #print('sector fields: ', sector)  
        df['is'+txt]=np.where(df[sector].sum(axis=1).gt(0), 1, 0)
    return(df)

def add_numerical_cols(data,more_data = False):
    # Creating variables about number of products sold:
    data['n_main_prod'] = data['meat'] + data['fruit'] + data['vegetables']
    data['n_other_prod'] = data['flowers'] + data['legumes'] + data['charcuterie']+ data['mushrooms'] + data['rice'] +\
        data['flour_cereals'] + data['oil_olives_vinager'] + data['eggs'] + data['dairies'] +\
        data['herbs_spices'] + data['hygiene_medicines'] + data['alcohol'] +\
        data['fruit_veggies_products'] + data['drinks'] + data['bread_pastries'] +\
        data['pasta'] + data['others']

    data['n_tot_prod'] = data['n_main_prod'] + data['n_other_prod']

    if more_data==True:
        # Creating variable about number of payment methods:
        data['n_paym_methods']=data.paym_bizum+data.paym_cash+data.paym_card+data.paym_transf 

        # Creating variable about number of comarcas where they deliver:
        data['n_comarcas_delivery']=data['DONDE'].apply(lambda x: x.count(',')+1 if 'Catalunya' not in x else 42)
    return(data)

def run_project_match(txt, df_col):
    '''Check if the name of the project/brand is in both datasets.
    The match is done checking both the ratio and partial ratio (per word).
    A "score" is associated to the match: 1 == exact match, 0.5 ==
    partial match'''
    n=0
    dupl = ''
    for i,row in df_col.iteritems():
        rat      = fuzz.ratio(txt.lower(),row.lower())
        part_rat = fuzz.partial_ratio(txt.lower(),row.lower())
        if 100 > rat > 70 and 100 > part_rat > 80:
            print('Partial match:', txt,'vs.',row,rat,part_rat)
            dupl = row
            n+=0.5
            break
        if rat == 100 and part_rat == 100:
            dupl = row
            n+=1
            break
    return str(n)+';'+txt+';'+dupl

def get_project_matches(col1,col2,not_duplicates=None):
    matches = col1.apply(lambda row: run_project_match(row, col2))
    matches = matches.str.split(';',expand=True).rename(columns={0:'exact',1:'marca_inicial',2:'MARCA'})

    # keep the values that are duplicates only
    matches = matches[matches.exact != '0']

    # remove false positives
    if not_duplicates != None:
        to_remove = matches[matches['marca_inicial'].str.contains(r'\b'+r'\b|\b'.join(not_duplicates)+r'\b')]
        print('Matches considered non-matches:\n',to_remove[['marca_inicial','MARCA']])
        ind = to_remove.index
        matches = matches.drop(ind)

    return(matches)

# ANALYSIS FUNCTIONS

def get_comarca_coords(data):
    geolocator = Nominatim(user_agent="my-app")
    country ="Spain"
    coord = []
    for comarca in [x for x in data['comarca_origin'].unique() if x not in ['(NOTFOUND)','Repartim al Bages, Solsonès, Barcelonès i Berguedà']]:
        #geolocator = Nominatim(user_agent="my-application")
        loc = geolocator.geocode(comarca+','+ country)
        coord.append({'comarca':comarca,'latitude':loc.latitude,'longitude':loc.longitude})
        time.sleep(1) #to avoid time out
    com_coord = pd.DataFrame(coord)
    return(com_coord)

def add_payment_types(data,pagament):
    for k,v in pagament.items():
        if k not in data.columns:
            data[k] = 0
    data.loc[(data.paym_card == 1) & (data.paym_cash == 1),'card+cash'] = 1
    data.loc[(data.paym_trans == 1) & (data.paym_cash == 1),'trans+cash'] = 1
    data.loc[(data.paym_trans == 1) & (data.paym_card == 1),'card+trans'] = 1
    data.loc[(data.paym_bizum == 1) & (data.paym_card == 1),'card+bizum'] = 1
    data.loc[(data.paym_bizum == 1) & (data.paym_cash == 1),'cash+bizum'] = 1
    data.loc[(data.paym_trans == 1) & (data.paym_bizum == 1),'trans+bizum'] = 1
    return(data)

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


def create_df_for_sankey(data):
    ##-- Connections between comarcas
    ##-- Creating the dataframe needed for sankey diagram (i.e. the list of all the edges between two comarcas),
    ##-- it will have the following columns: source, target, value.
    
    ## Extracting all the target comarcas from the field 'DONDE'
    df=[]

    for j in range(0,data.shape[0]):

        targets=data.DONDE.iloc[j].split(", ")
        n_targets=len(targets)
    
        if(n_targets>=40):
            df.append((data.comarca_origin.iloc[j], 'Catalunya', 1))
        else:
            for i in range(0,n_targets):
                df.append((data.comarca_origin.iloc[j], targets[i], 1))
    
    df = pd.DataFrame(df, columns=('source', 'target', 'value'))
    
    ## Removing records which have no info in the target or in the source field
    df=df[~(df.target=='')]
    df=df[~(df.source=='')]
    
    ## Uniformizing names of comarcas between the pagesos dataset and the cataloninan comarcas dataset (comarcas_df)
    standard_names = {
        'Osona / Lluçanès':'Osona', 
        'Ribera d’Ebre': 'Ribera d\'Ebre', 
        'Pla de l’Estany':'Pla de l\'Estany', 
        'Pla d’Urgell': 'Pla d\'Urgell',
        'Al Urgell':'Alt Urgell',
        'Bages-Moianès':'Moianès',
        'Moianes-Bages':'Moianès',
        'Barcelona':'Barcelonès',
        'Maresme-Barcelonès':'Maresme',
        'Tarragona':'Tarragonès',
        'Baix Montseny':'Vallès Oriental',
        'Baixa Cerdanya':'Cerdanya',
        'Vall Aran':"Vall d'Aran",
        'Alt Maresme':'Maresme',
        'Penedès':'Alt Penedès',
        "Val D'Aran": "Vall d'Aran",
        'Lluçanès':'Osona', #should we consider it a comarca?
        "La Seu d'Urgell": 'Alt Urgell',
        'El Vendrell': 'Baix Penedès',
        'Baix Llobregates':'Baix Llobregat'
    }
    
    df['target'] = df['target'].replace(standard_names)
    df['source'] = df['source'].replace(standard_names)
    
    
    ##-- Creation of the final df by grouping by (source, target) couples
    
    ## Getting the normalization factor (i.e. the total number of connections per comarca of origin)  
    df_norm=df.groupby(['source'])['value'] \
                             .sum() \
                             .reset_index(name='norm_factor') 

    ## Grouping by the connections with same source-target: 
    df_edges=df.groupby(['source', 'target'])['value'] \
                             .sum() \
                             .reset_index(name='value') \
                             .sort_values(['value'], ascending=False) \

    ## Adding the normalized factor to the edges df:
    df_edges=pd.merge(df_edges, df_norm, how='inner', left_on='source', right_on='source')
    df_edges['norm_value']=df_edges['value'].astype(float)/df_edges['norm_factor'].astype(float)*100

    return(df_edges)