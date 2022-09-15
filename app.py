import streamlit as st
import pandas as pd
import PIL
from joblib import dump, load

from bokeh.models.widgets import Div

import main_app


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


st.markdown("# Call to Action: Email Industry")



stats_col1, stats_col2, stats_col3, stats_col4 = st.columns([1,1,1,1])

with stats_col1:
    st.metric(label="Production", value="Development")
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
    'click_to_open_rate',
    'conversion_rate'
]

call_2_action = [ 
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

if st.button('Generate Predictions'):
    if uploaded_file is None:
        st.error('Please upload a email (HTML format)')
    else:
        placeholder = st.empty()
        placeholder.text('Loading Data')

        # Starting predictions
        model = load('models/CTA.joblib')
        