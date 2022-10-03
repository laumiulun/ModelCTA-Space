import streamlit as st
import pandas as pd
import PIL
import ipywidgets 
from joblib import dump, load

from bokeh.models.widgets import Div

import main_app

import utils

def table_data():
    # creating table data
    field = [
        'Data Scientist',
        'Dataset',
        'Algorithm',
        'Framework',
        'Ensemble',
        'Domain',
        'Model Size'
    ]

    data = [
        'Buwani',
        'Internal + Campaign monitor',
        'Random Forest',
        'Sci-kit learn',
        'Bootstrapping',
        'Bootstrapping Aggregation',
        '60.3 KB'
    ]

    data = {
        'Field':field,
        'Data':data
    }

    df = pd.DataFrame.from_dict(data)

    return df


def url_button(button_name,url):
    if st.button(button_name):
        js = """window.open('{url}')""".format(url=url) # New tab or window
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)


if 'generate_pred' not in st.session_state:
	st.session_state.generate_pred = False

st.markdown("# Call to Action: Email Industry")



stats_col1, stats_col2, stats_col3, stats_col4 = st.columns([1,1,1,1])

with stats_col1:
    st.metric(label="Production", value="Devel")
with stats_col2:
    st.metric(label="Accuracy", value="80.49%")

with stats_col3:
    st.metric(label="Speed", value="0.004 ms")

with stats_col4:
    st.metric(label="Industry", value="Email")


with st.sidebar:

    with st.expander('Model Description', expanded=False):
        img = PIL.Image.open("figures/ModelCTA.png")
        st.image(img)
        st.markdown('This model aims to provide email campaign services and campaign engineers with a greater understanding of how to format your Call-To-Action (CTA) features, trained on a large corpus of email campaign CTA successes and failures. This model provides real-time predictive analytics recommendations to suggest optimal CTAs focusing the users attention to the right text and color of your CTA content. The Loxz Digital CTA Feature Selection will provide the best way to send out campaigns without the opportunity cost and time lapse of A/B testing. Email metrics are provided prior to campaign launch and determine the optimal engagement rate based on several factors, including several inputs by the campaign engineer.')

    with st.expander('Model Information', expanded=False):
        # Hide roww index
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(table_data())

    url_button('Model Homepage','https://www.loxz.com/#/models/CTA')
    # url_button('Full Report','https://resources.loxz.com/reports/realtime-ml-character-count-model')
    url_button('Amazon Market Place','https://aws.amazon.com/marketplace')


industry_lists = [
    'Retail',
    'Software and Technology',
    'Hospitality',
    'Academic and Education',
    'Healthcare',
    'Energy',
    'Real Estate',
    'Entertainment',
    'Finance and Banking'
]

campaign_types = [
    'Promotional', 
    'Transactional', 
    'Webinar', 
    'Survey', 
    'Newsletter', 
    'Engagement',
    'Curated_Content', 
    'Review_Request', 
    'Product_Announcement', 
    'Abandoned_Cart'
]

target_variables = [
    'Click_To_Open_Rate',
    'Conversion_Rate'
]

call2action = [ 
    'Color','Text','Both'
]


uploaded_file = st.file_uploader("Please upload your email (In HTML Format)", type=["html"])

# if uploaded_file is None:
    # upload_img = PIL.Image.open(uploaded_file)
    # upload_img = None
# else:
    # upload_img = None


industry = st.selectbox(
    'Please select your industry',
    industry_lists
)

campaign  = st.selectbox(
    'Please select your industry',
    campaign_types
)

target = st.selectbox(
    'Please select your target variable',
    target_variables
)

call2action_feature = st.selectbox(
    'Select the Call-To-Action Feature you would like to analyze for predictive analytics',
    call2action
)

def generate_cta_list(num_text):
    cta_list = []
    for i in range(num_text):
        cta_list.append('CTA Number {}'.format(i+1))
    cta_list.append('All')
    return cta_list


def display_CTA(text,color):
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
            value="{}">""".format(i+1,color[i],text[i])
        if i != len(text)-1:
            base_string += "<br>"
    return base_string

generate_pred = st.button('Generate Predictions')
if generate_pred:
    st.session_state.generate_pred = True

if uploaded_file is None and st.session_state.generate_pred:
    st.error('Please upload a email (HTML format)')
elif uploaded_file is not None and st.session_state.generate_pred:
    placeholder = st.empty()
    placeholder.text('Loading Data')

    # Starting predictions
    vtext, ccolor, text = utils.email_parser(uploaded_file.getvalue().decode("utf-8"))

    if (len(ccolor) > 0) and (len(text) > 0):
        st.info("Number of Call-To-Actions in the email: {}".format(len(text)))
        cta_list = generate_cta_list(len(text))
        cta_selected = st.radio(
            'Select the Call-To-Action you would like to analyze ?',  
                cta_list)
        base_string = display_CTA(text,ccolor)
        st.components.v1.html(base_string,height = len(text)*30+50)

        predict = st.button('Predict Optimial CTA')


        cta_menu = []
        for i in range(len(text)):
            cta_menu.append(ipywidgets.Checkbox(
                value=False,
                description='Call-To-Action Text: {}'.format(i+1),
                disabled=False,
                indent=False
            ))
        if cta_selected == 'All':
            for i in range(len(text)):
                cta_menu[i].value = True
        else:
            index = int(cta_selected.split(' ')[-1])
            cta_menu[index].value = True


        if st.session_state.generate_pred and predict:
            utils.get_predictions(
                target,
                industry,
                campaign,
                call2action_feature,
                vtext,
                ccolor,
                text,
                cta_menu)

    else:
        st.error("The email you uploaded does not contain any Call-To-Actions.")
    

    placeholder.text('')

