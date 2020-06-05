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

def deliver_to(txt, locations, delivery_patt):
    # obtain the delivery locations given free text, 
    # a list of possible locations to look for
    # and patterns that indicate delivery specifications
    
    locations_in_text = ''
    # we look for sentences that look like delivery specifications
    patt_to_search = '(?:'+'|'.join(delivery_patt)+').*' 
    if re.search(patt_to_search,txt):
        relevant_txt = re.search(patt_to_search,txt).group().strip()
        for loc in locations:
            if re.search(r'\b'+loc+r'\b',relevant_txt):
                if loc not in locations_in_text:
                    locations_in_text = locations_in_text + ','+loc
        return locations_in_text.strip(',')

def get_text_locations(df, output_col, input_col, locations, delivery_patt):
    df[output_col] = df[input_col].apply(lambda row: 
                                              deliver_to(row, locations, delivery_patt))