import re
import time
import pandas as pd
from  geopy.geocoders import Nominatim
import preprocess as prep


def all_comarca_column_prep(pagesos, abastiment, data_gen, locations_df, stopwords, com_typos):
    '''Prepare "comarca_origin" column extracting data from existing columns, removing 
    typos and cheching spelling'''
    pagesos['comarca_origin']    = pagesos['MUNICIPIO'].str.split(')').str.get(-2).str.split('(').str.get(1).fillna('')
    pagesos['DONDE'] = pagesos['DONDE'].replace(com_typos,regex=True)
    pagesos['comarca_origin'] = pagesos['comarca_origin'].replace(com_typos,regex=True)
    abastiment['comarca_origin'] = abastiment['COMARCA'].replace(com_typos,regex=True)
    data_gen['comarca_origin']   = data_gen['Comarca'].str.title().replace(com_typos,regex=True)

    pagesos['comarca_origin']    = pagesos['comarca_origin'].apply(lambda x: check_comarca_spelling(
        x,locations_df['Comarca'],stopwords) if x not in locations_df['Comarca'] else x)
    abastiment['comarca_origin'] = abastiment['comarca_origin'].apply(lambda x: check_comarca_spelling(
        x,locations_df['Comarca'],stopwords) if x not in locations_df['Comarca'] else x)
    data_gen['comarca_origin']   = data_gen['comarca_origin'].apply(lambda x: check_comarca_spelling(
        x,locations_df['Comarca'],stopwords) if x not in locations_df['Comarca'] else x)

    return(pagesos,abastiment,data_gen)

def check_comarca_spelling(df_text,comarca_df,stopwords):
    '''Homogenize the comarca spelling data.
    Compare the comarca spelling from the dataframe with the one
    from the locations data (both preprocessed). If the dataframe one 
    is found, the value from the locations data will be return.
    Useful to recognize comarcas without the correct accentuation
    missing determinants...'''
    comarca_df = comarca_df.unique()
    coms_prep  = [prep.pre_process(x,stopwords,sw=True) for x in comarca_df]
    loc_prep   = prep.pre_process(df_text, stopwords,sw=True)
    if loc_prep in coms_prep:
        ind = coms_prep.index(loc_prep)
        return(comarca_df[ind])
    else:
        if df_text != '':
            print('\tComarca not found:',df_text)
        return('(NOTFOUND)')


def run_text_locations(data, locations_df, cols, delivery_patt):
    '''Apply the function to obtain the locations from free text fields'''
    for data_field in cols:
        for loc in ['Comarca','Capital','Provincia','Municipi']:
            get_text_locations(data,loc.lower(),data_field,locations_df,loc,delivery_patt)

def get_text_locations(df, output_col,col_to_search, locations_df, look_for, delivery_patt):
    '''Apply deliver_to function to run through the registers of the specified 
    column and look for locations'''
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


def abastiment_create_donde_col(data,mun_to_com_dict):
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


def get_comarca_coords(data):
    '''Get the coordinates of comarques in Catalonia using geopy'''
    data['comarca_origin'] = data['comarca_origin'].str.replace(r'^Urgell','Baix Urgell')
    geolocator = Nominatim(user_agent="my-app")
    country ="Spain"
    coord = []
    for comarca in [x for x in data['comarca_origin'].unique() if x not in ['(NOTFOUND)','Repartim al Bages, Solsonès, Barcelonès i Berguedà']]:
        #geolocator = Nominatim(user_agent="my-application")
        loc = geolocator.geocode(comarca+','+ country)
        coord.append({'comarca':comarca,'latitude':loc.latitude,'longitude':loc.longitude})
        time.sleep(1) #to avoid time out
    com_coord = pd.DataFrame(coord)
    return(data, com_coord)
