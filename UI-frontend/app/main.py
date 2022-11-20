import streamlit as st
import streamlit.components.v1 as components
# from PIL import Image
import requests
import pickle as pkl
import re
import pandas as pd
import json
import PyPDF2
import os
import itertools

from examples import EXAMPLES
from ner_examples import NER_EXAMPLES
from annotated_text import annotated_text
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np

# pip install st-annotated-text
# pip install wordcloud
# pip install Pillow

try:
    import StringIO
except ImportError:
    from io import StringIO

def get_topics(url, data):
    r = requests.post(url, json = data)
    print(r.status_code)
    if r.status_code != 200:
        return None
    else:
        return r.json()

def get_ner(url, data):
    r_ner = requests.post(url, json = data)
    print(r_ner.status_code)
    if r_ner.status_code != 200:
        return None
    else:
        return r_ner.json()
# def download_link(
#     content, label="Download", filename="file.txt", mimetype="text/plain"


# a fuction used for word cloud
def plot_wordcloud(list_topic:list) -> None:
    '''
    Plot a wordcloud of top 20 words from the input text
    masked by world logo
    '''
    text = " ".join(word for word in list_topic)
    # Create stopword
    stopwords = set(STOPWORDS)

    # mask = np.array(Image.open('/home/ked/client/img/worldmap1.png')) 
    wordcloud = WordCloud( 
        background_color='white', 
        stopwords=stopwords,
        random_state=42,
        max_words=20, 
        max_font_size=80).generate(text)
    # plt.figure(figsize=(10,10))
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.imshow(wordcloud, interpolation = "bilinear")
    plt.axis("off")
    st.pyplot(fig)



url = 'http://0.0.0.0:8000/topic/predict'
url_ner = 'http://0.0.0.0:9000/ner/predict'


st.set_page_config(page_title="GlgCapstone-Demo", page_icon=":star:", layout="wide")

st.subheader("GLG Topic Modeling and Named Entity Recognation")

tab1, tab2 = st.tabs(["Topic Analysis", "Named Entity Recognition"])

with tab1: 
    with st.expander("ℹ️ About Topic Model", expanded=True):

        st.write(
            """     
        -   Topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. 
        -   Topic models can help to organize and offer insights for us to understand large collections of unstructured text bodies.
            """
        )

    st.markdown("")

    
# st.set_page_config(layout="wide")

# st.markdown("""
# <style>
# .big-font {
#     font-size:300px !important;
# }
# </style>
# """, unsafe_allow_html=True)

# st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)


    # with st.form(key="my_form"):
    col1, col2, col3 = st.columns([3,1,3])

    with col1:
        prompts = list(EXAMPLES.keys()) + ["Select a document"]
        prompt = st.selectbox(
            'Example Inputs',
            prompts,
            index=2
        )

        if prompt == "Select a document":
            prompt_box = ""
        else:
            prompt_box = EXAMPLES[prompt]


    with col3:

        uploaded_file = col3.file_uploader("Upload pdf document", type=".pdf")
        if uploaded_file:
            # creating a pdf file object
            # pdfFileObj = StringIO(uploaded_file.getvalue().decode("utf-8"))
                
            # creating a pdf reader object
            pdfReader = PyPDF2.PdfFileReader(uploaded_file)
                
            # printing number of pages in pdf file
            print(pdfReader.numPages)
                
            # creating a page object
            pageObj = pdfReader.getPage(0)
            prompt_box = pageObj.extractText()
            # closing the pdf file object
            # pdfFileObj.close()

    doc_txt = st.text_area(
            "Document:",
            prompt_box, height=200
        )
    submit_button = st.button(label="Generate topics")


    # if not doc_txt:
    #     st.stop() # pop up message


    if submit_button:
        if doc_txt != "" and len(doc_txt.split(" ")) > 12:
            with st.spinner("Generating topics..."):
                data = {"document": {"0": doc_txt}}
                topics = get_topics(url, data)

                st.markdown("Model Output")

                tab1_result, tab2_result = st.tabs(["Result Tables", "Result Wordcloud" ])

                st.header("")
                df_global = pd.DataFrame(topics['topics']['0']['global'].items())
                df_global['label'], df_global['topics'] = df_global[1].apply(lambda x: x['labels']), df_global[1].apply(lambda x: x['topics'])
                df_global = df_global.set_index(df_global[0])
                df_global.drop(1, axis=1, inplace=True)

                df_local = pd.DataFrame(topics['topics']['0']['local'].items())
                df_local['label'], df_local['topics'] = df_local[1].apply(lambda x: x['labels']), df_local[1].apply(lambda x: x['topics'])
                df_local = df_local.set_index(df_local[0])
                df_local.drop(1, axis=1, inplace=True)

                with tab1_result:

                    st.header("Global Topics")

                    st.table(df_global)

                    st.header("Local Topics")

                    st.table(df_local)

                with tab2_result:
                    global_topics = df_global['topics'].tolist()
                    global_labels = df_global['label'].tolist()
                    local_topics  = df_local['topics'].tolist()
                    local_labels  = df_local['label'].tolist()
                    global_topic_label = global_topics + global_labels
                    local_topic_label = local_topics + local_labels
                    col4, col_, col5 = st.columns([2,1,2])

                    with col4:
                        st.header("Global Topics as a wordcloud")
                        plot_wordcloud(global_topic_label)


                    with col5:

                        st.header("Local Topics as a wordcloud")
                        plot_wordcloud(local_topic_label)
        else:
            st.warning('Please insert a document', icon="⚠️")
with tab2: 
    with st.expander("ℹ️ Named Entity Recognition", expanded=True):

        st.write(
            """     
            Named Entity Recognition is the task of identifying named entities (people, locations, organizations, etc.) in the input text.

            """
        )

    tab3, tab4 = st.tabs(["Demo", "Model Info"])

    with tab3:
        prompts_ner = list(NER_EXAMPLES.keys()) + ["Select a Sentence"]
        prompt_ner = st.selectbox(
            'Example Document',
            prompts_ner,
            index=3
        )

        if prompt_ner == "Select a Sentence":
            prompt_box = ""
        else:
            prompt_box = NER_EXAMPLES[prompt_ner]

        sent_txt = st.text_area(
        "Sentence:",
        prompt_box, height=100)
        submit_button_ner = st.button(label="Run Model")
        if submit_button_ner:
            if sent_txt != "":
                with st.spinner("Generating entities..."):
                    sent_data = {"sentence": sent_txt}
                    ner_output = get_ner(url_ner, sent_data)

                    st.markdown("Model Output")
                    st.markdown("Entities")
                    tokens_ner = ner_output['ner_tags']['tokens'][1:-1]
                    labels_ner = ner_output['ner_tags']['labels'][1:-1]

                    print(zip(tokens_ner, labels_ner))
                    annotated_list = []
                    ner_entities = ['per','gpe','geo','art','eve','org','tim','nat']

                    for i,token_label in enumerate(zip(tokens_ner, labels_ner)):
                        token, label = token_label[0], token_label[1]
                        if label.lower() not in ['o', 'pad']:
                            tag = label.split("-")
                            if tag[0] == "B":
                                collector = token
                                flag = True
                                j = i+1
                                while flag:
                                    if labels_ner[j].lower() not in ['o', 'pad']:
                                        if labels_ner[j].split("-")[1] != tag[1]:
                                            flag = False
                                        else:
                                            collector = collector + " " +tokens_ner[j]
                                            j += 1
                                    else:
                                        flag = False
                                annotated_list.append((collector, tag[1]))
                        else:
                            annotated_list.append(token+" ")
                    print(annotated_list)
                    # st.write(annotated_list)
                    annotated_text(*annotated_list)
    with tab4:
        data_path = os.getcwd()
        os.path.join(data_path,"data/modelcard.csv")
        df = pd.read_csv(os.path.join(data_path,"data/modelcard.csv"), sep=',')
        # df = df.rename(columns={'0':'','1':''})

        st.table(df)

