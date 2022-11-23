<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Forthbrain NLP Capstone Project (GLG Topic Modeling and Named Entity Recognation)
</h3>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-describtion">Project Describtion</a>
    </li>
     <li>
      <a href="#project-objective">Project Objective</a>
    </li>
    <li>
      <a href="#built-with">Built With</a>
    </li>
     <li>
      <a href="#data-source">Data Source</a>
    </li>
    <li>
      <a href="#topic-modeling-pipeline">Topic Modeling Pipeline</a>
      <ul>
        <li><a href="#data-cleaning">Data Cleaning and Data Exploration</a></li>
        <li><a href="#document-Embedding">Document Embedding</a></li>
        <li><a href="#feature-reduction">Feature Reduction</a></li>
        <li><a href="#document-clustering">Document Clustering</a></li>
        <li><a href="#topic-representation">Topic Representation</a></li>
      </ul>
    </li>
    <li>
      <a href="#named-entity-recognation">Named Entity Recognation</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation with Docker-compose</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#support">Support</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>



<!--PROJECT Describtion-->
## Project Describtion

<p align="justify">
GLG is the world’s insight network that connects decision makers to a network of experts so they can act with the confidence that comes from true clarity and have what it takes to get ahead. GLG receives a large amount of health and tech requests from clients seeking insights on topics of different domains. Preprocessing client requests and extracting relevant topic/keyword detection take extra time and need a large manpower. This project on Natural Language Processing (NLP) is aimed at improving the topic/keyword detection process from the client submitted reports and identifying the underlying patterns in submitted requests over time. The primary challenges include Named Entity Recognition (NER) and Pattern Recognition for Hierarchical Clustering of Topics.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--PROJECT Objective-->
## Project Objective

<p align="justify">

The purpose of this project is to develop an NLP model capable of recognizing and clustering topics related to technological and healthcare terms given a large text corpus in an unsupervised manner and to develop a Named Entity Recognition model capable of extracting entities from a given sentence.

</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--Built With-->
## Built With

* Python
      <ul>
        <li>NumPy/pandas</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
        <li>Keras</li>
        <li>Pythorch</li>
        <li>Seaborn</li>
        <li>Streamlit</li>
      </ul>
* Language Models
      <ul>
        <li>SBERT</li>
        <li>NLTK</li>
      </ul>
* Jupyter Notebook
* Visual Studio Code

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--Data Source-->
## Data Source

<p align="justify">

* [All the News 2.0](https://components.one/datasets/all-the-news-2-news-articles-dataset/) — This dataset contains 2,688,878 news articles and essays from 27 American publications, spanning January 1, 2016 to April 2, 2020.

* [Annotated Corpus for NER](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) — Annotated Corpus for Named Entity Recognition using GMB(Groningen Meaning Bank) corpus for entity classification with enhanced and popular features by Natural Language Processing applied to the data set.

</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--Topic Modeling Pipeline-->
## Topic Modeling Pipeline

![alt text](https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering/blob/main/doc/topic_modeling_pipeline.png)

<p align="justify">
Topic models are useful tools to discover latent topics in collections of documents. In this section below,  we look into details of the various parts of the topic modeling pipeline with highlights and key findings. 
</p>

<p align="justify">
<strong>Data Cleaning and Data Exploration:</strong> The first step in the pipeline is data cleaning and Data Exploration of news article dataset. From the original data we extract the news articles that focus only on the health and technology section. Then we performed different kinds of text data cleaning steps like:
      <ul>
        <li>punctuation and non-alphanumeric character removal.</li>
        <li>Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.</li>
        <li>Words that have fewer than 3 characters are removed.</li>
        <li>All stopwords are removed.</li>
        <li>Words are lemmatized</li>
      </ul>
</p>

<p align="justify">
<strong>Document Embedding:</strong> We embed documents to create representations in vector space that can be compared semantically. We assume that documents containing the same topic are semantically similar. To perform the embedding step, first we extract a sentences in each document using NLTK sentence tokenizer and we apply the [Sentence-BERT (SBERT) framework](https://arxiv.org/abs/1908.10084) in each sentence and generate vector representation for each sentence, finally we represent a single document using dot product of each sentence vector representation and generate an embedding vector for the document. These embeddings, however, are primarily used to cluster semantically similar documents and not directly used in generating the topics.
</p>

<p align="justify">
<strong>Feature Reduction</strong>: In the above document embedding step, we embed each document using SBERT which generates a 768 long dense vector. Working with such a high dimension vector is computationally heavy and complex, hence, we apply dimensionality reduction technique called UMAP([Uniform Manifold Approximation and Projection](http://arxiv.org/abs/1802.03426)) to reduce the number of features/vectors without losing important information. 
</p>

<p align="justify">
<strong>Document clustering</strong>:  Finally we apply the [HDBSCAN](https://www.theoj.org/joss-papers/joss.00205/10.21105.joss.00205.pdf) (Hierarchical density based clustering) algorithm in order to extract clusters of semantically similar documents. It is an ex-tension of DBSCAN that finds clusters of varying densities by converting DBSCAN into a hierarchi-cal clustering algorithm. HDBSCAN models clusters using a soft-clustering approach allowing noise to be modeled as outliers. This prevents unrelated documents from being assigned to any cluster and is expected to improve topic representations.
</p>

<p align="justify">
<strong>Topic Representation</strong>: The topic representations are modeled based on the documents in each cluster where each cluster will be assigned more than one Global and Local topics. Using HDBSCAN algorithm we access Hierarchical structure of the documents in each cluster. This means in each cluster the documents distributed as parent and child hierarchical structure. Therefore, for each cluster we can extract Global and Local topics by applying the [LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) (Latent Dirichlet Allocation) model on those documents. Thus, we have 2 LDA Models for each cluster responsible to generate Global and Local topics for parent and child documents respectively.
</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!--Named Entity Recognation-->
## Named Entity Recognation

<p align="justify">

* NER is a widely used NLP technique that recognizes entities contained in a piece of text, commonly things like people organization, locations etc. This project also includes an NER model implemented using [BERT](https://arxiv.org/abs/1810.04805) and huggingface PyTorch library to quickly and efficiently fine-tune the BERT model to do the state of the art performance in Named Entity Recognition. The transformer package provides a BertForTokenClassification class for token-level predictions. BertForTokenClassification is a fine-tuning model that wraps BertModel and adds a token-level classifier on top of the BertModel. The token-level classifier is a linear layer that takes as input the last hidden state of the sequence. 

* Below is an example of an input and output of our named entity model, served with a streamlit app.

</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
1. Install Git lfs for installation guide [see tutorial](https://git-lfs.github.com)
2. Install Docker for installation guide [see tutorial](https://docs.docker.com/get-docker/)
3. Install Docker Compose for installation guide [see tutorial](https://docker-docs.netlify.app/compose/install/#upgrading)

### Installation with Docker-compose

To package the whole solution which uses multiple images/containers, I used Docker Compose. Please follow the steps below for successful installation.

1. Clone the repo
   ```sh
   git lfs clone https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering.git
   ```
2. Go to the project directory
   ```sh
   cd GLG--Topic-Modeling-and-Document-Clustering
   ```
3. Create a bridge network
   Since we have multiple containers communicating with each other, I created a bridge network called AIservice. 
   First create the network AIService by running this command:
   ```sh
   docker network create AIservice
   ```
4. Run the whole application by executing this command:
   ```sh
   docker-compose up -d --build
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

<strong>Frontend app with Streamlit</strong>

You can see the frontend app in the browser using : http://localhost:8501/ or
If you are launching the app in the cloud, replace localhost with your public Ip address. 

![alt text](https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering/blob/main/doc/frontend_main.png)

Please refer to this Documentation for more.

_For more examples, please refer to the [Documentation](https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering/tree/main/doc)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Support -->
## Support

Contributions, issues, and feature requests are welcome!

Give a ⭐️ if you like this project!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

* <strong>Kedir Ahmed</strong> - [@linkedin](https://www.linkedin.com/in/kedir-ahmed/) - [kedirhamid@gmail.com](kedirhamid@gmail.com)
* <strong>Ranganai Gwati</strong> - [ranganaigwati@gmail.com](ranganaigwati@gmail.com)
* <strong>Aklilu Gebremichail</strong> - [akliluet@gmail.com](akliluet@gmail.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- References -->
## References

* [Sentence-bert: Sentence embeddings using siamese bert-networks.](https://arxiv.org/abs/1908.10084)
* [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](http://arxiv.org/abs/1802.03426)
* [Hierarchical density based clustering](https://www.theoj.org/joss-papers/joss.00205/10.21105.joss.00205.pdf)
* [BERT:Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794)
* [Named Entity Recognation with BERT](https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/)
* [Best-Readme-Template](https://github.com/othneildrew/Best-README-Template#getting-started)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/kedir/GLG--Topic-Modeling-and-Document-Clustering.svg?style=for-the-badge
[forks-url]: https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering/network/members
[stars-shield]: https://img.shields.io/github/stars/kedir/GLG--Topic-Modeling-and-Document-Clustering.svg?style=for-the-badge
[stars-url]: https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering/stargazers
[issues-shield]: https://img.shields.io/github/issues/kedir/GLG--Topic-Modeling-and-Document-Clustering.svg?style=for-the-badge
[issues-url]: https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering/issues
[license-shield]: https://img.shields.io/github/license/kedir/GLG--Topic-Modeling-and-Document-Clustering.svg?style=for-the-badge
[license-url]: https://github.com/kedir/GLG--Topic-Modeling-and-Document-Clustering/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kedirahmed
