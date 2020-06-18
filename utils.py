import re
import unicodedata

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
        return(loc_prep)

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