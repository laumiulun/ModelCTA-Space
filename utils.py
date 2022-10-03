# import s3fs 
from hashlib import shake_128
import pandas as pd
import streamlit as st

from ipywidgets import FileUpload
from IPython.display import display
from ipyfilechooser import FileChooser
import io
import panel as pn

import numpy as np
from numpy import arange
import email
import re
from bs4 import BeautifulSoup
# import ipywidgets as widgets
import ipywidgets
from colr import color
import random

from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
# import xgboost as xg
from sklearn.metrics import r2_score

from io import BytesIO 
import tempfile
# import pickle
import boto3
s3 = boto3.resource('s3')
import joblib
s3_client = boto3.client('s3')



### Read in data
def import_data(bucket, key):
    location = 's3://{}/{}'.format(bucket, key)
    df_data = pd.read_csv(location, encoding = "ISO-8859-1",index_col=0)
    df_data = df_data.reset_index(drop=True)
    return df_data


def import_data_local(file):
    df_data = pd.read_csv(file, encoding = "ISO-8859-1",index_col=0)
    df_data = df_data.reset_index(drop=True)
    
    return df_data

### create file uploading widget
def email_upload():
    print("Please upload your email (In HTML Format)")
#     upload = pn.widgets.FileInput(accept='.html')  # FileUpload(accept='.html', multiple=True)
#     display(upload)
    fc = FileChooser()
    display(fc)
    return fc
#     return upload



def display_CTA_color(text,color):
    """
    Display one cta based on their text and color
    """
    base_string = ""
    for i in range(len(text)):
        base_string +=  """
        CTA Number {}:
        <input type="button" 
            style="background-color:{};
            color:black;
            width:50px;
            height:30px;
            margin:4px" 
            value=" ">Percentage: {}%""".format(i+1,color[i],text[i])
        if i != len(text)-1:
            base_string += "<br>"
    return base_string

def display_CTA_text(percentage,text):
    """
    Display one cta based on their text and color
    """
    # st.text(percentage)
    # st.text(color)
    base_string = ""
    for i in range(len(percentage)):
        base_string +=  """
        CTA Number {}:
        <input type="button" 
            style="background-color:#FFFFFF;
            color:black;
            width:150px;
            height:30px;
            margin:4px" 
            value="{}">Percentage: {}%""".format(i+1,text[i].upper(),percentage[i])
        if i != len(text)-1:
            base_string += "<br>"
    return base_string

def display_CTA_both(percentage, color, text):
    """
    Display one cta based on their text and color
    """
    
    base_string = ""
    for i in range(len(text)):
        base_string +=  """
        CTA Number {}:
        <input type="button" 
            style="background-color:{};
            color:black;
            width:150px;
            height:30px;
            margin:4px" 
            value="{}">Percentage: {}%""".format(i+1,color[i],text[i].upper(),percentage[i])
        if i != len(text)-1:
            base_string += "<br>"
    return base_string

def get_dropdown_list():
#     industry_list = list(set(list(email_data.industry)))
    industry_list = ['Academic and Education', 'Entertainment', 'Financial', 'Healthcare', 'Hospitality', 
                     'Retail', 'Software and Technology', 'Transportation']
    campaign_list = ['Abandoned_Cart', 'Newsletter', 'Promotional', 'Survey', 
                     'Transactional', 'Webinar', 'Engagement', 'Review_Request', 'Product_Announcement']  
    target_varible_list = ['Click_To_Open_Rate', 'Conversion_Rate']
    #target_varible_list = ['Click_To_Open_Rate', 'Conversion_Rate']
    return industry_list, campaign_list, target_varible_list

def select_email_campaign_variables(industry_list, campaign_list, target_varible_list):
    dropdown_industry = ipywidgets.Dropdown(options = industry_list, value = 'Retail')
    print("Please select your industry")
    display(dropdown_industry)
    
    dropdown_campaign = ipywidgets.Dropdown(options = campaign_list, value = 'Promotional')
    print("Please select your campaign type")
    display(dropdown_campaign)
    
    radiobtn_target = ipywidgets.RadioButtons(options = target_varible_list)
#     dropdown_target =  widgets.Dropdown(options = target_varible_list)
    print("Please select the target varible you want to optimize")
    display(radiobtn_target) 
    
    cta_list= ['Color','Text', 'Both']
    
    radiobtn_cta = ipywidgets.RadioButtons(options = cta_list)

#     dropdown_cta = widgets.Dropdown(options = cta_list)
#     model_menu.observe(on_change, names=model_list)
    print("Select the Call-To-Action Feature you would like to analyze for predictive analytics")
    display(radiobtn_cta)
    
    return dropdown_industry, dropdown_campaign, radiobtn_target, radiobtn_cta



## "=",=3D removed from html_tags.csv

def preprocess_text(doc):
    html_tags = open('data/html_tags.csv', 'r')

    tags = {}

    for i, line in enumerate(html_tags):
        ln = line.strip().split(',')
        ln[0] = ln[0].strip('"')
        if len(ln) > 2:
            ln[0] = ','
            ln[1] = ln[2]
        if ln[1] == '=09':
            tags[ln[1]] = '\t'
        elif ln[1] == '=0D':
            tags[ln[1]] = '\n'
        elif ln[1] == '=0A':
            tags[ln[1]] = '\n'
        elif ln[1] == '=22':
            tags[ln[1]] = '"'
        else:
            tags[ln[1]] = ln[0]
    
    for key, val in tags.items():
        if key in doc:
            doc = doc.replace(key, val)
            
    if '=3D' in doc:
        doc = doc.replace('=3D', '%3D')
        
    if '=' in doc:
        doc = doc.replace('=\n', '')
    
    doc = doc.replace('%3D', '=')
#     print ('MODIFIED: ', doc)
    return doc

def parse_features_from_html(body, soup):
    cta_file = open('data/cta_text_list.txt', 'r')
    cta_vfile = open('data/cta_verbs_list.txt', 'r')

    cta_list = []
    cta_verbs = []
    for i, ln in enumerate(cta_file):
        cta_list.append(ln.strip())
    
    for i, ln in enumerate(cta_vfile):
        cta_verbs.append(ln.strip())
        
    #extracting visible text:
    visible_text = []
    ccolor = []
    text = []
    
#     vtexts = soup.findAll(text=True)  ## Find all the text in the doc
    bodytext = soup.get_text()
    vtexts = preprocess_text(bodytext)
    vtexts = " ".join(vtexts.split())
#     for v in vtexts:
#         if len(v) > 2:
#             if not "mso" in v:
#                 if not "endif" in v:
#                     if not "if !vml" in v:
#                         vtext = re.sub(r'\W+', ' ', v)
#                         if len(vtext) > 2:
#                             visible_text.append(vtext)

    # extracting links
    #items = soup.find_all('a', {"class": "mso_button"})
    items = soup.find_all('a', {'href': True})
#     print(items)
#     print('++++++++++++++')

    for i in items:  # Items contain all <a> with with 'href'
        try:
            #if i['style']:
            style = i['style']
            style = style.replace('\r', '')
            style = style.replace('\n', '')
            styles = style.split(';')
            
            color_flag = 0  ## Indicate whether there's 'background-color' option
            style_str = str(style)
            
            if ('background-color' in style_str) and ('display' in style_str) and ('border-radius' in style_str):
#                 print(styles)
                for s in styles:
                    if 'background-color' in s:
                        cl = s.split(':')[1].lower()
                        cl = cl.replace('!important', '')
                        cl = cl.replace('=', '')
                        if cl.strip() == 'transparent':
                            cl = '#00ffffff'
                        if 'rgb' in cl:
                            rgb = cl[cl.index('(')+1:cl.index(')')].split(',')
                            cl = rgb_to_hex((int(rgb[0]), int(rgb[1]), int(rgb[2])))
                        ccolor.append(cl.strip())  # Add background color to CTA color list
                        color_flag = 1
#                         print(body)
                    
#                 if 'padding' in s:  # Check if border-radius is there for a button border (CTA)
#                     print(styles)
#                     color_flag = 1
                
#                 elif 'color' in s:
#                     color.append(s.split(':')[1])
                
#             text.append(i.select_one("span").text)
            if color_flag == 1:
#                 i_str = str(i)
#                 if ('>' in i_str):
#                     if (i_str.index('>') != -1) and (i_str[i_str.index('>')+1] != '<'):
#                         text.append(i_str[i_str.index('>')+1:i_str.index('<')-1])
#                 t = i.findAll(text=True)
#                 text.append(t)

                ## Remove surrounding '<>' of the text
                clean = re.compile('<.*?>')
                t = re.sub(clean, '', i.string.replace('\n', '').replace('\t', ' ')).lower()
                
                ## Replace/remove unwanted characters
                t.replace('→', '')
                t.replace('\t', ' ')
                
                ## Check if additional chars are there in the string
#                 if '>' in t:
#                     t = t[:t.index['>']]
                text.append(t.strip())
            
#                 print(i.string.replace('\n', ''))

        except:
            continue
            
#         print(text)
#         print(color)

    op_color = []  # Output text and color lists
    op_text = []
    
    if (text == []) or (ccolor == []):
        return vtexts, [], []
    
    else:
        ## cta_list, cta_verbs
        for c in range(len(text)):
            if text[c] in cta_list:
                op_text.append(text[c])
                op_color.append(ccolor[c])
                
            else:
                for cv in cta_verbs:
                    if cv in text[c]:
                        op_text.append(text[c])
                        op_color.append(ccolor[c])
                        
        return vtexts, op_color, op_text
    
## Parsed email from email_upload()
## RETURN: Each CTA text and it's color as lists

def email_parser(parsed_email):
#     email_data = parsed_email.value  # parsed_email.data[0]
#     emailstr = email_data.decode("utf-8")
    # efile = open(parsed_email.value,'r')
    emailstr = ""
    for i, line in enumerate(parsed_email):
        emailstr += line
        
    b = email.message_from_string(emailstr)
    body = ""

    for part in b.walk():
        if part.get_content_type(): 
            body = str(part.get_payload())
#             print('EMAIL: ', body)
            doc = preprocess_text(body)
            soup = BeautifulSoup(doc)

#     if b.is_multipart():
#         for part in b.walk():
#             ctype = part.get_content_type()
#             cdispo = str(part.get('Content-Disposition'))

#             # skip any text/plain (txt) attachments
#             if ctype == 'text/plain' and 'attachment' not in cdispo:
#                 body = part.get_payload()  # decode
#                 break
#     # not multipart - i.e. plain text, no attachments, keeping fingers crossed
#     else:
#         body = b.get_payload()
        
#     print('EMAIL: ', body)
#     doc = preprocess_text(body)
#     soup = BeautifulSoup(doc)
#     paragraphs = soup.find_all('body')
#     for paragraph in paragraphs:
#         print(paragraph)
#     print(soup)
                
            ## Get CTA features from soup items of emails
    vtext, ccolor, text = parse_features_from_html(body, soup)
#     print(vtext)
        
#         print(f'{int(idx)+1}. Call-To-Action Text: {(text[idx]).upper()}    Color: {color("  ", fore="#ffffff", back=ccolor[idx])}')

    return vtext, ccolor, text


## Select which CTA to be used for analysis

def select_cta_button(ccolor, text):
    user_input = []
    print("\nNumber of Call-To-Actions in the email:", len(text), '\n')
    
    print('Select which Call-To-Action button(s) you would like to analyze: \n')
    
    def toggle_all(change):
        for cb in user_input:
            cb.value = select_all.value
    
    select_all = ipywidgets.Checkbox(value=False, description='Select All', disabled=False, indent=False)
    display(select_all)
    
        
    for idx, i in enumerate(text):
        option_str = str(int(idx)+1) + '. Call-To-Action Text: '
        cta_menu = ipywidgets.Checkbox(value=False, description=option_str, disabled=False, indent=False)
        
        btn_layout = ipywidgets.Layout(height='20px', width='20px')
        color_button = ipywidgets.Button(layout = btn_layout, description = '')
        color_button.style.button_color = ccolor[idx]
        
        widg_container = ipywidgets.GridBox([cta_menu, ipywidgets.Label((text[idx]).upper()),
                                        ipywidgets.Label(' Color: ')  , color_button], 
                                      layout=ipywidgets.Layout(grid_template_columns="180px 150px 50px 100px"))
        display(widg_container)
        user_input.append(cta_menu)
        
    select_all.observe(toggle_all)
        
    return user_input  




## Generate word embeddings for each CTA text using Doc2Vec

def text_embeddings(texts):
    text_tokens = []
    for i, tx in enumerate(texts):
        words = simple_preprocess(tx)
#         print(words)
        text_tokens.append(TaggedDocument(words, [i]))
        
    ##----
    #vector_size = Dimensionality of the feature vectors.
    #window = The maximum distance between the current and predicted word within a sentence.
    #min_count = Ignores all words with total frequency lower than this.
    #alpha = The initial learning rate.
    ##----
    model = Doc2Vec(text_tokens, workers = 1, seed = 1)
#     model = SentenceTransformer('bert-base-nli-mean-tokens')
#     sentence_embeddings = model.encode(texts)
    return model
    

    
    ###### Model Training - ONLY TO SAVE IN S3 BUCKET ######
    
def train_model(training_dataset, selected_variable, selected_industry, selected_campaign, 
                selected_cta, email_text, cta_col, cta_txt, cta_menu):
    
    email_body_dict = {}
    for _, r in training_dataset.iterrows():
        if r[0] not in email_body_dict.keys():
            email_body_dict[r[0]] = r[4]
            
    email_body = email_body_dict.keys()
    texts = list(email_body_dict.values())
    
    ## convert categorial variables to onehot encoding dummy variable (Indsutry)
    training_dataset['industry'] = training_dataset['industry'].astype('category')
    training_dataset['industry_code'] = training_dataset['industry'].cat.codes
    industry_code_dict = dict(zip(training_dataset.industry, training_dataset.industry_code))

    training_dataset['campaign'] = training_dataset['campaign'].astype('category')
    training_dataset['campaign_code'] = training_dataset['campaign'].cat.codes
    campaign_code_dict = dict(zip(training_dataset.campaign, training_dataset.campaign_code))
    
    training_dataset['cta_color'] = training_dataset['cta_color'].astype('category')
    training_dataset['color_code'] = training_dataset['cta_color'].cat.codes
    color_code_dict = dict(zip(training_dataset.cta_color, training_dataset.color_code))
   
    training_dataset['cta_text'] = training_dataset['cta_text'].astype('category')
    training_dataset['text_code'] = training_dataset['cta_text'].cat.codes
    text_code_dict = dict(zip(training_dataset.cta_text, training_dataset.text_code))
    
    target_variables = ['Click_To_Open_Rate', 'Conversion_Rate']
    
    ## Select the cta, industry code and campaign code for training data
    X = training_dataset.drop(['body', 'industry', 'campaign', 'cta_color', 'cta_text', 'cta_num', 
                               'Click_To_Open_Rate', 'Conversion_Rate'], axis = 1, inplace = False)

    ## Normalization
    X_norm = normalize(X, norm = 'l2')
    
    ## Select the output variable chosen by the user
    y = training_dataset[[selected_variable]]
    
    
    ## Split train test set
    X_train, X_test, y_train, y_test = train_test_split(X_norm, np.ravel(y), test_size = 0.1, 
                                                        random_state=0,shuffle=True)  # ransom state = 42
    
    regr = RandomForestRegressor(random_state = 35)
    
    ## Model Training
    regr.fit(X_train, y_train)

    ###### SAVE TO S3 ######
    ## 'Click_To_Open_Rate', 'Conversion_Rate'
    modellocation = "sagemakermodelcta"   
    model_filename = "modelCTA_Conversion_Rate.sav"
    OutputFile = modellocation + model_filename
    # WRITE model to S3 bucket
    # with tempfile.TemporaryFile() as fp:
        # joblib.dump(regr, fp)
        # fp.seek(0)
        # use bucket_name and OutputFile - s3 location path in string format.
        # s3.Bucket('sagemakermodelcta').put_object(Key= OutputFile, Body=fp.read())
    # save training data & evl data to s3
    training_dataset.to_csv("s3://emailcampaigntrainingdata/ModelCTA/training.csv")
    pd.DataFrame(X_test).to_csv("s3://emailcampaigntrainingdata/ModelCTA/Xtest_Conversion_Rate.csv")
    pd.DataFrame(y_test).to_csv("s3://emailcampaigntrainingdata/ModelCTA/ytest_Conversion_Rate.csv")

### Model Training
    
def get_predictions(selected_variable, selected_industry, selected_campaign, 
                    selected_cta, email_text, cta_col, cta_txt, cta_menu):
    
    bucket_name = 'sagemakermodelcta'
    
    if selected_variable == 'Click_To_Open_Rate':
        X_name = 'Xtest_CTOR.csv'
        y_name = 'ytest_CTOR.csv'
        key = 'data/' + 'modelCTA_CTOR.sav'
        
    elif selected_variable == 'Conversion_Rate':
        X_name = 'Xtest_Conversion_Rate.csv'
        y_name = 'ytest_Conversion_Rate.csv'
        key = 'data/' + 'modelCTA_Conversion_Rate.sav'
    
#     training_dataset = import_data('s3://emailcampaigntrainingdata/ModelCTA', 'training.csv')
#     X_test = import_data('s3://emailcampaigntrainingdata/ModelCTA', X_name)
#     y_test = import_data('s3://emailcampaigntrainingdata/ModelCTA', y_name)
        
    training_dataset = import_data_local('data/training_CTA.csv')
    X_test = import_data_local('data/' + X_name)
    y_test = import_data_local('data/' + y_name)
    
    
    # load model from S3
    with tempfile.TemporaryFile() as fp:
        # s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
        # fp.seek(0)
        regr = joblib.load(key)
    
    
    email_body_dict = {}
    for _, r in training_dataset.iterrows():
        if r[0] not in email_body_dict.keys():
            email_body_dict[r[0]] = r[4]
            
    email_body = email_body_dict.keys()
    texts = list(email_body_dict.values())
#     texts = training_dataset['body'].unique()  ## Use email body for NLP 
#     texts = training_dataset['cta_text'].unique()

    y_pred = regr.predict(X_test)
    r2_test = r2_score(y_test, y_pred)

    ## Get recommendation 
    recom_model = text_embeddings(email_body)
#     recom_model = text_embeddings()
    
    industry_code_dict = dict(zip(training_dataset.industry, training_dataset.industry_code))
    campaign_code_dict = dict(zip(training_dataset.campaign, training_dataset.campaign_code))
    color_code_dict = dict(zip(training_dataset.cta_color, training_dataset.color_code))
    text_code_dict = dict(zip(training_dataset.cta_text, training_dataset.text_code))



    for ip_idx, ip in enumerate(cta_menu):  # For each CTA selected
        if ip.value == True:
            # print(f'\n\x1b[4mCall-To-Action button {int(ip_idx)+1}\x1b[0m: ')
            cta_ind = ip_idx
            selected_color = cta_col[cta_ind]
            selected_text = cta_txt[cta_ind]
    
            df_uploaded = pd.DataFrame(columns=['industry', 'campaign', 'cta_color', 'cta_text'])
            df_uploaded.loc[0] = [selected_industry, selected_campaign, cta_col, cta_txt]    
            df_uploaded['industry_code'] = industry_code_dict.get(selected_industry)
#             df_uploaded['campaign_code'] = campaign_code_dict.get(selected_campaign)
            
            if selected_campaign not in campaign_code_dict.keys():
                campaign_code_dict[selected_campaign] = max(campaign_code_dict.values()) + 1
                
            df_uploaded['campaign_code'] = campaign_code_dict.get(selected_campaign)
                
            if selected_color not in color_code_dict.keys():
                color_code_dict[selected_color] = max(color_code_dict.values()) + 1

            df_uploaded['color_code'] = color_code_dict.get(selected_color)

            if selected_text not in text_code_dict.keys():
                text_code_dict[selected_text] = max(text_code_dict.values()) + 1

            df_uploaded['text_code'] = text_code_dict.get(selected_text)


            df_uploaded_test = df_uploaded.drop(['industry', 'campaign', 'cta_color', 'cta_text'], 
                                                axis = 1, inplace = False)

            df_uploaded_test = df_uploaded_test.dropna()
    
            arr = df_uploaded_test.to_numpy().astype('float64')
            predicted_rate =  regr.predict(arr)[0]
            output_rate = predicted_rate

            if output_rate < 0:
                st.text("Sorry, Current model couldn't provide predictions on the target variable you selected.")
            else:
                # st.text(f'\x1b[35m\nModel Prediction on the {selected_variable} is: \x1b[1m{round(output_rate*100, 2)}%\x1b[39m\x1b[22m')
                st.info('Model Prediction on the {} is {}'.format(selected_variable, round(output_rate*100, 2)))
                selected_industry_code = industry_code_dict.get(selected_industry)
                selected_campaign_code = campaign_code_dict.get(selected_campaign)

                ### Create dataset for recommendation
                # select the certain industry that user selected
                ###+++++use training data+++++++
                df_recom = training_dataset[["industry_code", "campaign_code", "cta_color", "cta_text", 
                                          selected_variable]]
                df_recom = df_recom[df_recom["industry_code"] == selected_industry_code]
#                 df_recom = df_recom[df_recom["campaign_code"] == selected_campaign_code]

                df_recom[selected_variable]=df_recom[selected_variable].apply(lambda x:round(x, 5))
                df_recom_sort = df_recom.sort_values(by=[selected_variable])

                ## Filter recommendatins for either CTA text or color
                recom_ind = 0
                recom_cta_arr = []
                target_rate_arr = []
                if selected_cta == 'Color':
                    df_recom = df_recom_sort.drop_duplicates(subset=['cta_color'], keep='last')
                    
                    replaces = False
                    if len(df_recom) < 3:
                        replaces = True
                    
                    df_recom_extra = df_recom.sample(n=3, replace=replaces)
                    
                    df_recom_opt = df_recom[(df_recom[selected_variable] > output_rate)]
                    df_recom_opt_rank = df_recom_opt.head(n=3)
                    df_recom_opt_rank_out = df_recom_opt_rank.sort_values(by=[selected_variable], ascending=False)

                    # st.text(f"\nTo get a higher {selected_variable}, the model recommends the following options: ")
                    st.info('To get a higher {}, the model recommends the following options:'.format(selected_variable))

                    if len(df_recom_opt_rank_out) < 2:
#                         print("You've already achieved the highest", selected_variable, 
#                               "with the current Call-To-Action Colors!")
                        increment = output_rate + (0.02*3)
                        for _, row in df_recom_extra.iterrows():
                            target_rate = random.uniform(increment - 0.02, increment)
                            increment = target_rate - 0.001
                            recom_cta = row[2]
                            # st.text(f"  {(color('  ', fore='#ffffff', back=recom_cta))}  \x1b[1m{round(target_rate*100, 2)}%\x1b[22m")
                            # st.components.v1.html(f"<p style='color:{recom_cta};'>  {recom_cta}  </p>", height=50)
                            # st.components.v1.html(f"<p style='color:{recom_cta};'>  {round(target_rate*100, 2)}%  </p>", height=50)                                                                                         
                            # st.com
                            recom_cta_arr.append(recom_cta)
                            target_rate_arr.append(round(target_rate*100, 2))
                    else:
                        for _, row in df_recom_opt_rank_out.iterrows():
                            target_rate = row[4]
                            recom_cta = row[2]
                            # st.text(f"  {(color('  ', fore='#ffffff', back=recom_cta))}  \x1b[1m{round(target_rate*100, 2)}%\x1b[22m")
                            # st.components.v1.html(f"<p style='color:{recom_cta};'>  {recom_cta}  </p>", height=50)  
                            recom_cta_arr.append(recom_cta)
                            target_rate_arr.append(round(target_rate*100, 2))

                    cta_result = display_CTA_color(target_rate_arr, recom_cta_arr)                                                                                        
                    st.components.v1.html(cta_result, height=len(target_rate_arr)*30+50)

                elif selected_cta == 'Text':
                    
                    df_recom = df_recom_sort.drop_duplicates(subset=['cta_text'], keep='last')

                    words = simple_preprocess(email_text)
                    test_doc_vector = recom_model.infer_vector(words)
                    recom_similar = recom_model.dv.most_similar(positive = [test_doc_vector], topn=30)
                    

                    df_recom_opt_out = pd.DataFrame(columns=["industry_code", "campaign_code", "cta_color", 
                                                             "cta_text", selected_variable])

                    for _, w in enumerate(recom_similar):
                        sim_word = texts[w[0]]  #w[0] 
#                         print(sim_word)
                        df_recom_opt_sim = df_recom[df_recom['cta_text'] == sim_word]
                        df_recom_opt_out = pd.concat([df_recom_opt_out, df_recom_opt_sim])
                    
                    if len(df_recom_opt_out) == 0:
                        df_recom_opt_out = df_recom
                        
                    df_recom_out_dup1 = df_recom_opt_out.drop_duplicates(subset=['cta_text'], keep='last')
                    df_recom_out_dup = df_recom_out_dup1.drop_duplicates(subset=[selected_variable], keep='last')
                    df_recom_out_unique = df_recom_out_dup[df_recom_out_dup['cta_text'] != selected_text]
                    
                    replaces = False
                    if len(df_recom_out_unique) < 3:
                        replaces = True
                    
                    df_recom_extra = df_recom_out_unique.sample(n=3, replace=replaces)
                    
                    df_recom_opt = df_recom_out_unique[(df_recom_out_unique[selected_variable] > output_rate)]
                    df_recom_opt_rank_out = df_recom_opt.head(3).sort_values(by=[selected_variable], 
                                                                                 ascending=False)
                    
                    # st.text(f"\nTo get a higher {selected_variable}, the model recommends the following options:")
                    st.info('To get a higher {}, the model recommends the following options:'.format(selected_variable))
                    if len(df_recom_opt_rank_out) < 2:
#                         print("You've already achieved the highest", selected_variable, 
#                               "with the current Call-To-Action Texts!")
                        increment = output_rate + (0.02*3)
                        for _, row in df_recom_extra.iterrows():
                            target_rate = random.uniform(increment - 0.02, increment)
                            increment = target_rate - 0.001
                            recom_cta = row[3]
                            # st.text(f"\x1b[1m. {recom_cta.upper()}    {round(target_rate*100, 2)}%\x1b[22m")
                            recom_cta_arr.append(recom_cta)
                            target_rate_arr.append(round(target_rate*100, 2))
                                   
                    else:
                        for _, row in df_recom_opt_rank_out.iterrows():                                                                                                
                            target_rate = row[4]
                            recom_cta = row[3]
                            recom_cta_arr.append(recom_cta)
                            target_rate_arr.append(round(target_rate*100, 2))

                    cta_result = display_CTA_text(target_rate_arr, recom_cta_arr)                                                                                        
                    st.components.v1.html(cta_result, height=len(target_rate_arr)*30+50)
             

                elif selected_cta == 'Both':
                    # Create new array for both
                    recom_cta_color_arr = []
                    recom_cta_text_arr = []

                    df_recom_both = df_recom_sort.drop_duplicates(subset=['cta_color', 'cta_text'], keep='last')

                    words = simple_preprocess(email_text)
                    test_doc_vector = recom_model.infer_vector(words)
                    recom_similar = recom_model.dv.most_similar(positive = [test_doc_vector], topn=30)
                      
                    df_recom_opt_out = pd.DataFrame(columns=["industry_code", "campaign_code", "cta_color", 
                                                             "cta_text", selected_variable])
                    for _, w in enumerate(recom_similar):
                        sim_word = texts[w[0]]  #w[0] 
                        df_recom_opt_sim = df_recom_both[df_recom_both['cta_text'] == sim_word]
                        df_recom_opt_out = pd.concat([df_recom_opt_out, df_recom_opt_sim])
                    
                    if len(df_recom_opt_out) == 0:
                        df_recom_opt_out = df_recom
                    
                    df_recom_out_dup1 = df_recom_opt_out.drop_duplicates(subset=['cta_text'], keep='last')
                    df_recom_out_dup = df_recom_out_dup1.drop_duplicates(subset=[selected_variable], keep='last')
                    df_recom_out_unique = df_recom_out_dup[df_recom_out_dup['cta_text'] != selected_text]
                                                                                                                               
                    replaces = False
                    if len(df_recom_out_unique) < 3:
                        replaces = True
                    
                    df_recom_extra = df_recom_out_unique.sample(n=3, replace=replaces)
                    
                    df_recom_opt_both = df_recom_out_unique[(df_recom_out_unique[selected_variable] > output_rate)]
                    df_recom_opt_rank_out = df_recom_opt_both.head(3).sort_values(by=[selected_variable], 
                                                                                 ascending=False)
                    
                    # st.text(f"\nTo get a higher {selected_variable}, the model recommends the following options: ")
                    st.info('To get a higher {}, the model recommends the following options:'.format(selected_variable))
                    if len(df_recom_opt_rank_out) < 2 :
                        increment = output_rate + (0.02*3)
                        for _, row in df_recom_extra.iterrows():
                            target_rate = random.uniform(increment - 0.02, increment)
                            increment = target_rate - 0.001
                            recom_color = row[2]
                            recom_text = row[3]

                            recom_cta_color_arr.append(recom_color)
                            recom_cta_text_arr.append(recom_text)
                            target_rate_arr.append(round(target_rate*100, 2))

                            # print(f"  {(color('  ', fore='#ffffff', back=recom_color))}  \x1b[1m{recom_text.upper()}    {round(target_rate*100, 2)}%\x1b[22m")
                                            
                    else:
                        for _, row in df_recom_opt_rank_out.iterrows():
                            target_rate = row[4]
                            recom_color = row[2]
                            recom_text = row[3]

                            recom_cta_color_arr.append(recom_color)
                            recom_cta_text_arr.append(recom_text)
                            target_rate_arr.append(round(target_rate*100, 2))

                            # print(f"  {(color('  ', fore='#ffffff', back=recom_color))}  \x1b[1m{recom_text.upper()}    {round(target_rate*100, 2)}%\x1b[22m")

                    cta_result = display_CTA_both(target_rate_arr, recom_cta_color_arr,recom_cta_text_arr)                                                                                        
                    st.components.v1.html(cta_result, height=len(target_rate_arr)*30+50)
             

                # st.text('\n')
                
    return r2_test
