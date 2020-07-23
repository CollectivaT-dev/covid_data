import pandas as pd
import numpy as np
import os
import yaml
import json
import re
import unicodedata
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from stop_words import get_stop_words
from pathlib import Path # reads paths in the current OS
from PIL import Image
import location_utils as loc

def read_yaml(path, filename):
    filename = filename+'.yaml'
    with open(Path(path) / filename) as file:
        f = yaml.full_load(file)
    return(f)

def check_output_files():
    paths = read_paths_data()
    if os.path.isdir(paths['output']):
        coord = os.path.isfile(Path(paths['output']) / 'comarca_coordinates.csv')
        covid = os.path.isfile(Path(paths['output']) / 'covid_data.csv')
        gen   = os.path.isfile(Path(paths['output']) / 'vdp_clean.csv')
        if coord and covid and gen:
            execute = input("It looks like you have all the necessary files for the analysis.\n\
Do you want to execute the process anyway and update them? (y = yes, n = no): ")
            while execute not in ['y','n']:
                execute = input("Please enter 'y' for yes or 'n' for no: ")
        else:
            execute = 'y'
    else:
        os.mkdir(paths['output'])
        execute = 'y'
    return(execute)


def check_input_files():
    paths      = read_paths_data()
    pagesos    = os.path.isfile(Path(paths['input']) / 'db_mesinfo.json')
    abastiment = os.path.isfile(Path(paths['input']) / 'abastiment.csv')
    data_gen   = os.path.isfile(Path(paths['input']) / 'Productors_adherits_a_la_venda_de_proximitat.csv')

    if not all([pagesos, abastiment, data_gen]):
        print('It looks like some input files are missing.\n\
Check that in your input folder are present:\n\
db_mesinfo.json, abastiment.csv, Productors_adherits_a_la_venda_de_proximitat.csv')
        input_f = False
    else:
        input_f = True
    return(input_f)


def read_initial_data():
    
    conf  = read_yaml('conf', 'conf')
    paths = read_paths_data()

    pagesos      = pd.read_json(Path(paths['input']) / 'db_mesinfo.json', orient='index').fillna('').replace('’',"'", regex=True)
    abastiment   = pd.read_csv(Path(paths['input']) / "abastiment.csv", sep=",").fillna('').replace('’',"'", regex=True)
    data_gen = pd.read_csv(Path(paths['input']
                           ) / 'Productors_adherits_a_la_venda_de_proximitat.csv').fillna('').replace('’',"'", regex=True)

    data_gen.rename(columns   = {'Marca Comercial':'MARCA'}, inplace=True)
    abastiment.rename(columns = {'PROJECTE':'MARCA'}, inplace=True)
    pagesos.rename(columns    = {'url':'URL', 
                                 'Nom de la persona productora:':'PRODUCTOR', 
                                 'Marca:':'MARCA', 
                                 'Municipi:':'MUNICIPIO',  
                                 'On serveix:':'DONDE', 
                                 'Productes disponibles:':'PRODUCTOS', 
                                 'Altres productes alimentaris:':'OTROS',
                                 'Possibilitats pagament:': 'PAGO', 
                                 'Fruita*:':'FRUTA', 
                                 '*':'NOTAS', 
                                 'Més informació:':'INFO', 
                                 'Carn:':'CARNE', 
                                 'Verdura*:': 'VERDURA', 
                                 'Flor i planta ornamental:':'FLORES'}, inplace=True)

    for col in ['COMARCA','MUNICIPI']:
        abastiment[col] = abastiment[col].str.title()

    pagesos['dataset']    = 'pagesos'
    abastiment['dataset'] = 'abastiment'

    locations_df = pd.read_csv(Path(paths['input']) / 'municipis_merge.csv').fillna('')
    for col in locations_df.columns:
        locations_df[col] = locations_df[col].str.title()

    stopwords = get_all_stopwords()
    return(pagesos,abastiment,data_gen,locations_df,stopwords,paths,conf)

def read_paths_data():
    paths = read_yaml('conf','paths')
    return(paths)

def get_all_stopwords():
    '''Get list with catalan stopwords'''
    stopwords = get_stop_words('catalan')
    #add new stopwords
    new_stopwords = ['que','des', 'al', 'del', 'ho', 'd', 'l','per','tambe', 'fins',
                   'a', 'cap', 'hi', 'ni', 'no']
    stopwords.extend(new_stopwords)
    return(stopwords)
    

def all_prep_dataset(pagesos, abastiment, data_gen, locations_df, stopwords):
    '''Apply the pre-process to each dataframe specifying which columns to apply it to'''
    pagesos      = run_preprocess_on_cols(pagesos,['OTROS','PAGO'],stopwords)
    data_gen     = run_preprocess_on_cols(data_gen,['Productes','Grups Productes'],stopwords)
    abastiment   = run_preprocess_on_cols(abastiment,['COM COMPRAR','OBSERVACIONS',
        'PRODUCTE(S)','comarca_origin'],stopwords)
    locations_df = run_preprocess_on_cols(locations_df,
        ['Municipi','Comarca','Capital','Provincia'],stopwords)

    pagesos = pagesos_food(pagesos)
    return(pagesos, abastiment, data_gen)

def run_preprocess_on_cols(data,cols,stopwords):
    '''Apply the pre-process function to each row for every input column
    returning the results in a new column in the same dataframe'''
    for col in cols:
        data[col+'_prep'] = data[col].apply(lambda x: pre_process(x,stopwords,sw=True))
    return(data)

def pre_process(text, stopwords=[], sw=False):
    '''Pre-process functions to apply to a text'''
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    ## remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)

    # remove accents
    text=re.sub("[\u0300-\u036f]", "", unicodedata.normalize('NFD', text))
    text=text.strip()
    
    if sw:
        stopwords = [re.sub("[\u0300-\u036f]", "", 
               unicodedata.normalize('NFD', x)) for x in stopwords]
        text = re.sub(r'\s+',' ',re.sub(
            r'\b'+r'\b|\b'.join(stopwords)+r'\b','',text)).strip()
    return text

def pagesos_food(pagesos):
    '''Updating the columns vegetables, fruit, meat and flowers, 
    since in this dataset are informed in different fields (and not 
    in the main one OTROS)'''
    pagesos.loc[pagesos.VERDURA != '', 'vegetables']=1
    pagesos.loc[pagesos.FRUTA   != '', 'fruit']     =1
    pagesos.loc[pagesos.CARNE   != '', 'meat']      =1
    pagesos.loc[pagesos.FLORES  != '', 'flowers']   =1
    cols = ['vegetables','fruit','meat','flowers']
    pagesos[cols] = pagesos[cols].replace(np.nan, 0)
    return(pagesos)


def abastiment_payment_col(abastiment, payment_dict):
    # PAYMENT COLUMN
    abastiment = get_payment_methods(abastiment,['COM COMPRAR', 'OBSERVACIONS','PRODUCTE(S)'],
        payment_dict)
    return(abastiment)

def get_payment_methods(data,imp_cols,payment_dict):
    '''Creates a new column containing a list of the different payment
    methods found in the free text fields'''
    data['PAGO_prep'] = ''
    for c in [x+'_prep' for x in imp_cols]:
        for k,v in payment_dict.items():
            paym_word  = pre_process(v)
            data_pay_w = data[data[c].str.contains(r'\b'+paym_word+r'\b')]
            if data_pay_w.shape[0]!=0:
                ind = data_pay_w.index
                data.loc[ind,'PAGO_prep'] = data.loc[ind]['PAGO_prep'] +','+ paym_word
    data['PAGO_prep'] = data['PAGO_prep'].str.strip(',')
    return(data)


def all_add_new_cols(pagesos, abastiment, data_gen, locations_df, conf):
    '''Apply the functions to obtain new columns for the dataframes
    related with payment method, locations and binary variables'''

    # LOCATIONS COLUMN
    # Dictionary to translate municipis to comarca
    locations_df = locations_df[locations_df['Municipi']!='']
    
    loc.run_text_locations(abastiment, locations_df, ['COM COMPRAR', 'OBSERVACIONS','PRODUCTE(S)','comarca_origin'],
        conf['buying_method']['delivery'])
    locations_df = locations_df.assign(Municipi_reg=r'\b'+ locations_df['Municipi'] +r'\b')
    mun_to_com_dict = locations_df.set_index('Municipi_reg')['Comarca'].to_dict()
    abastiment = loc.abastiment_create_donde_col(abastiment,mun_to_com_dict)

    # BINARY COLUMNS
    already_pag_cols = ['meat','vegetables','fruit','flowers']
    pagesos_bin_prod = {key:value for key, value in conf['products'].items() if key not in already_pag_cols}
    # create binary variables representing whether they have a type of product (1) or not
    create_binary_var(pagesos,pagesos_bin_prod,'OTROS'+'_prep')
    create_binary_var(abastiment,conf['products'],'PRODUCTE(S)'+'_prep')
    create_binary_var(data_gen,conf['products'],'Productes'+'_prep')
    # create binary variables representing whether they have a payment method or not
    create_binary_var(pagesos,conf['payment'],'PAGO_prep')
    create_binary_var(abastiment,conf['payment'],'PAGO_prep')
    # create binary variables representing whether they have a type payment method, contact info... (1) or not
    create_binary_var(abastiment,conf['buying_method'],'COM COMPRAR'+'_prep')

    abastiment = abastiment_improve_binary_cols(abastiment, conf)

    pagesos    = create_sectors_col(pagesos,conf['sectors'])
    abastiment = create_sectors_col(abastiment,conf['sectors'])
    data_gen   = create_sectors_col(data_gen,conf['sectors'])

    pagesos    = add_payment_combis(pagesos,dict(conf['payment'], **conf['payment']))
    abastiment = add_payment_combis(abastiment,dict(conf['payment'], **conf['payment']))
    return(pagesos, abastiment, data_gen)

def create_binary_var(data,dic,col):
    '''Create a column with binary values based on if the input column has the 
    text contained in the values of the input dictionary or not'''
    for key, val in dic.items():
        data[key]=0
        if isinstance(val,list):
            val = [pre_process(v) for v in val]
            vals_to_look_for = r'\b'+r'\b|\b'.join(val)+r'\b'
        else:
            val = pre_process(val)
            vals_to_look_for = val
        data.loc[data[col].str.contains(vals_to_look_for),key] = 1

def abastiment_improve_binary_cols(abastiment, conf):
    '''Improve the website, social network and iseco columns for abastiment dataset'''
    web_words    = r'\b'+r'\b|\b'.join(conf['buying_method']['web'])+r'\b'
    social_words = r'\b'+r'\b|\b'.join(conf['buying_method']['socialnet'])+r'\b'

    abastiment.loc[(abastiment['web']!=1) & 
        (abastiment['OBSERVACIONS'+'_prep'].str.contains(web_words)),'web'] = 1
    abastiment.loc[(abastiment['socialnet']!=1) & 
        (abastiment['OBSERVACIONS'+'_prep'].str.contains(social_words)),'socialnet'] = 1

    abastiment.loc[(abastiment['iseco'] == 0) & 
        (abastiment['CCPAE'].isin(['Sí','En conversió'])),'iseco'] = 1
    return(abastiment)

def create_sectors_col(data,dic):
    '''Create 5 columns with binary values representing if a producer belong to a given sector or not, 
    as defined in the input dictionary (at least one column of the list of columns related to the sector must be 1)'''
    for sector, sector_cols in dic.items():
        if isinstance(sector_cols,list):
            data = data.assign(**{'is_'+sector:np.where(data[sector_cols].eq(1).any(axis=1),1,0)})
        else:
            data = data.assign(**{'is_'+sector:np.where(data[sector_cols].eq(1).any(),1,0)})
    return(data) 

def add_payment_combis(data,payment_combis):
    for k,v in payment_combis.items():
        if k not in data.columns:
            data[k] = 0
    data.loc[(data.bizum == 1) & (data.targeta == 1),'card+bizum'] = 1
    data.loc[(data.bizum == 1) & (data.efectiu == 1),'cash+bizum'] = 1
    data.loc[(data.targeta == 1) & (data.efectiu == 1),'card+cash'] = 1
    data.loc[(data['transferencia previa'] == 1) & (data.efectiu == 1),'trans+cash'] = 1
    data.loc[(data['transferencia previa'] == 1) & (data.targeta == 1),'card+trans'] = 1
    data.loc[(data['transferencia previa'] == 1) & (data.bizum == 1),'trans+bizum'] = 1
    return(data)


def all_add_num_cols(pagesos, abastiment, data_gen, conf):
    '''Apply the computation of numerical columns to each dataframe'''
    pagesos    = compute_numerical_cols(pagesos,conf,more_data=True)
    abastiment = compute_numerical_cols(abastiment,conf,more_data=True)
    data_gen   = compute_numerical_cols(data_gen,conf,more_data=False)
    abastiment.loc[abastiment['n_paym_methods']==0,'n_paym_methods'] = np.nan
    return(pagesos, abastiment, data_gen)

def compute_numerical_cols(data,conf,more_data = False):
    '''Adding some numerical columns to a dataframe'''
    # Creating variables about number of products sold:
    cols_main_products  = ['meat','vegetables','fruit']
    cols_other_products = [col for col in list(conf['products'].keys()
        ) if col not in cols_main_products+['iseco']]

    data['n_main_prod']  = data[cols_main_products].replace('', np.nan).sum(axis=1)
    data['n_other_prod'] = data[cols_other_products].replace('', np.nan).sum(axis=1)
    data['n_tot_prod']   = data['n_main_prod'] + data['n_other_prod']

    if more_data==True:
        cols_payment = list(conf['payment'].keys())
        # Creating variable about number of payment methods:
        data['n_paym_methods']=data[cols_payment].sum(axis=1)
        # Creating variable about number of comarcas where they deliver:
        data['n_comarcas_delivery']=data['DONDE'].apply(lambda x: x.count(',')+1 if 'Catalunya' not in x else 42)
    return(data)


# MERGE DATA
def merge_covid_data(pagesos, abastiment, locations, conf):
    '''Apply get_project_matches to merge all covid data (pagesos and 
    abastiment) removing the duplicate brands so they appear only 
    once in the final dataframe'''
    common_cols = [c for c in pagesos.columns if c in abastiment.columns]

    matches = get_project_matches(pagesos['MARCA'],abastiment['MARCA'],
        conf['brand_not_duplicates']['covid'])
    covid_data = pd.concat([pagesos[common_cols],
                          abastiment[~(abastiment.MARCA.isin(matches.MARCA.values))][common_cols]],
                          axis=0)
    covid_data = covid_data.merge(locations[['Comarca','Provincia']].drop_duplicates(), 
                  left_on = 'comarca_origin',right_on='Comarca',how='left')
    return(covid_data)

def covid_in_gen(covid_data, data_gen, conf):
    '''Apply get_project_matches to find which names are in the dataset 
    from the generalitat'''
    vdp_matches = pd.DataFrame(columns=['exact','marca_inicial','MARCA'])

    for col in ['Nom productor','MARCA']:
        print('Searching matches in column:',col)
        matches = get_project_matches(data_gen['Nom productor'],covid_data['MARCA'],
            conf['brand_not_duplicates']['gen'][col])
        vdp_matches = pd.concat([vdp_matches, matches],axis=0)
        print("\t =================================\n\n")
        
    vdp_matches = vdp_matches.drop_duplicates()
    vdp_matches = vdp_matches.rename(columns={'marca_inicial':'marca_vdp'})
    print('\t Number of coincidences with other datasets:',vdp_matches.shape[0])

    covid_data = covid_data.merge(vdp_matches[['marca_vdp','MARCA']],on='MARCA',how='left')
    return(covid_data)

def get_project_matches(col1,col2,not_duplicates=None):
    '''Apply run_project_match to filter with fuzzy matching and 
    exact matching which brand names can be duplicated'''
    matches = col1.apply(lambda row: run_project_match(row, col2))
    matches = matches.str.split(';',expand=True).rename(columns={0:'exact',1:'marca_inicial',2:'MARCA'})

    # keep the values that are duplicates only
    matches = matches[matches.exact != '0']

    # remove false positives
    if not_duplicates != None:
        false_pos_to_look = r'\b'+r'\b|\b'.join(not_duplicates)+r'\b'
        to_remove = matches[matches['marca_inicial'].str.contains(false_pos_to_look)]
        print('Final matches after removing false positives:\n',
            matches[~matches['marca_inicial'].str.contains(false_pos_to_look)][
            ['marca_inicial','MARCA']].to_string(index=False))
        ind = to_remove.index
        matches = matches.drop(ind)

    return(matches)

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
            print('\t Partial:', txt,'vs.',row,'. Ratio:',rat,', partial ratio:',part_rat)
            dupl = row
            n+=0.5
            break
        if rat == 100 and part_rat == 100:
            dupl = row
            n+=1
            break
    return str(n)+';'+txt+';'+dupl


def save_merged_data(covid_data, data_gen, com_coord, paths):
    covid_data.to_csv(Path(paths['output']) / 'covid_data.csv', index=False)
    com_coord.to_csv(Path(paths['output']) / 'comarca_coordinates.csv', index=False)
    data_gen.drop(['Productes_prep', 'Grups Productes_prep'],axis=1
             ).to_csv(Path(paths['output']) / 'vdp_clean.csv', index=False)



# ANALYSIS
def read_final_data():
    conf       = read_yaml('conf','conf')
    paths      = read_paths_data()
    data_gen   = pd.read_csv(Path(paths['output']) / 'vdp_clean.csv').fillna('')

    covid_data = pd.read_csv(Path(paths['output']) / 'covid_data.csv').fillna('')
    com_coord  = pd.read_csv(Path(paths['output']) / 'comarca_coordinates.csv')
    cat = Image.open(Path(paths['input'])
                 /'mapa_comarques_catalunya.png')
    return(covid_data, data_gen, com_coord, conf, cat)
