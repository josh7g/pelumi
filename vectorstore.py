#General Imports
import json
import os
import sys
import boto3
import traceback
import numpy as np
import time
import random
import re
from datetime import date

## Bedrock Imports
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock, BedrockEmbeddings

## Data Ingestion Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFDirectoryLoader #load directory
from langchain_community.document_loaders import PyPDFLoader #load_pdf
from langchain_community.document_loaders import TextLoader #load_txt
from langchain_community.document_loaders import MongodbLoader

## Vector Embedding & Vector Store imports
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time

## Context Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## LLM Query Imports
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

## AST Parser
import trees_python

## File Splitting Chunker
import chunker

#--- Service Declarations ---#

##  Bedrock Clients
aws_region = os.environ.get('AWS_REGION', 'us-east-1')  # Default to us-east-1 if envir key is not set

bedrock = boto3.client("bedrock-runtime", region_name=aws_region)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

##  Pinecone Instantiation
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

use_serverless = True
pinecone_instance = Pinecone(api_key=pinecone_api_key)

if use_serverless:
    spec_instance = ServerlessSpec(cloud='aws', region=aws_region)
else:
    spec_instance = PodSpec()

##  Read configuration file
def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                name, value = line.strip().split('=', 1) # Split on the first '=' only
                config[name] = value
    return config


## ========== INSERTION START ========== ##


#--- Data Ingestion ---#


##  Read file in plaintext
def read_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            #print(content) #debug
            return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


##  File Loader for Data ingestion
def load_file(file_path):
    file_extension = file_path.rsplit('.', 1)[-1].lower()

    # Choose the loader based on the file format
    if file_extension == 'pdf':
        loader = PyPDFLoader(file_path) #pdfload
    elif file_extension == 'txt':
        loader = TextLoader(file_path) #textload
    elif file_extension == 'docx':
        print("Pelumi, call the right function for docx o")
        loader = DocxLoader(file_path) #docxload
    else:
        text_content = read_file(file_path)
        if text_content is None:
           raise ValueError("Unsupported file format")
        else:
           #write to text file or write logic to return string or loader and perform loader.load or direct text embedding in caller function
           loader = StringLoader(text_content)
    return loader.load()


##  Chunk Document
def data_ingestion_pinecone0(file_path):
    documents = load_file(file_path)
    text_splitter = SemanticChunker(bedrock_embeddings)

    # Timing the split_documents function
    start_time = time.time()
    docs = text_splitter.split_documents(documents)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to split text in this document: {elapsed_time:.2f} seconds")

    # Verify chunk sizes and details
    """print("Verifying initial chunking:")
    valid_docs = []
    for i, doc in enumerate(docs):
        if len(doc.page_content) < 1:
            print(f"Skipping empty chunk {i+1}")
            continue

        start_position = i * (6000 - 600)
        end_position = start_position + len(doc.page_content)
        print(f"Chunk {i+1} length: {len(doc.page_content)}, start: {start_position}, end: {end_position}")
        print(f"Chunk {i+1} content (first 100 chars): {doc.page_content[:100]}...")

        if i > 0 and len(valid_docs) > 0:  # Ensure there is a previous valid chunk
            prev_chunk_end = valid_docs[-1].page_content[-600:]
            current_chunk_start = doc.page_content[:600]
            #print(f"Overlap with previous chunk (first 100 chars): {current_chunk_start}")
            #print(f"Previous chunk overlap area (last 100 chars): {prev_chunk_end}")

        valid_docs.append(doc)

    return valid_docs"""


# Break the document down to sections < 5mb so the 8k context window of titan text embed isn't exceeded
def split_large_document(documents, max_size=5 * 1024 * 1024):
    doc_sections = []
    current_section = []
    current_size = 0

    for doc in documents:
        page_size = len(doc.page_content.encode('utf-8'))
        #print(f"Page size: {page_size} bytes")  # Debug print
        if current_size + page_size > max_size:
            print(f"Creating new section, current section size: {current_size} bytes")  # Debug print
            doc_sections.append(current_section)
            current_section = [doc]
            current_size = page_size
        else:
            current_section.append(doc)
            current_size += page_size

    if current_section:
        print(f"Final section size: {current_size} bytes")  # Debug print
        doc_sections.append(current_section)

    return doc_sections


# Chunk the documents
def process_sections(doc_sections):
    text_splitter = SemanticChunker(bedrock_embeddings)

    start_time = time.time()
    docs = text_splitter.split_documents(doc_sections)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to split documents: {elapsed_time:.2f} seconds")

    # Verify chunk sizes and details
    print("Verifying initial chunking:")
    valid_docs = []
    chunk_id = 0
    for i, doc in enumerate(docs):
        if len(doc.page_content) < 1:
            print(f"Skipping empty chunk {i+1}")
            continue

        start_position = i * (6000 - 600)
        end_position = start_position + len(doc.page_content)
        print(f"Chunk {i+1} length: {len(doc.page_content)}, start: {start_position}, end: {end_position}")
        print(f"Chunk {i+1} content (first 100 chars): {doc.page_content[:100]}...")

        if i > 0 and len(valid_docs) > 0:  # Ensure there is a previous valid chunk
            prev_chunk_end = valid_docs[-1].page_content[-600:]
            current_chunk_start = doc.page_content[:600]
            # print(f"Overlap with previous chunk (first 100 chars): {current_chunk_start}")
            # print(f"Previous chunk overlap area (last 100 chars): {prev_chunk_end}")

        valid_docs.append(doc)

    return valid_docs


##  Chunk doc
def data_ingestion_pinecone_from_docs(file_path, file_description=None):
    documents = load_file(file_path)
    # Split the large document into smaller sections
    doc_sections = split_large_document(documents)
    all_valid_docs = []
    total_sections = len(doc_sections)

    #for section in doc_sections:
    for i, section in enumerate(doc_sections):
        print(f"Processing section {i+1}/{total_sections} with {len(section)} pages...")
        valid_docs = process_sections(section)
        print(f"Section {i+1} processed. Valid chunks count: {len(valid_docs)}")

        all_valid_docs.extend(valid_docs)
        print(f"Total valid chunks so far: {len(all_valid_docs)}")

        # Calculate and print the progress
        progress = ((i + 1) / total_sections) * 100
        print(f"Progress: {progress:.2f}%")

    if file_description is not None:
        metadatas = all_valid_docs[0]
        metadatas.metadata['description'] = file_description

    #print(all_valid_docs)
    return all_valid_docs



##  Get raw data from DB
def mongo_load_app(userID): #customerID
    # Read the connection string from the config file
    config = read_config('config.txt')

    # Define the parameters
    connection_string = config.get('MONGODB_URL')
    db_name = "rezliant-staging"
    collection_name = "applications"
    filter_criteria = {"userId": userID}  # Optional
    field_names = ["name", "techstack", "description"]  # Optional

    # Create the MongoDBLoader instance
    loader = MongodbLoader(
        connection_string=connection_string,
        db_name=db_name,
        collection_name=collection_name,
        filter_criteria=filter_criteria,
        field_names=field_names
    )

    documents = loader.load()
    return documents


##  Summarize data from DB
def app_summarize(documents):
    app_summary = None
    app_name = "bare"
    # Separate field values
    for doc in documents:
        doc_content = doc.page_content

        # Regular expressions to capture the patterns
        str_pattern = r"([^\[]+)" # Extract the first string
        list_pattern = r"\[([^\]]+)\]"  # Extract the list
        second_str_pattern = r"\] ([^\[]+)$"  # Extract the second string

        # Extract the first string
        match_str = re.search(str_pattern, doc_content)
        #app_name = match_str.group(1).strip() if match_str else None
        time.sleep(5)
        app_name = match_str.group(1).strip() if match_str else "bare"

        # Extract the list
        match_list = re.search(list_pattern, doc_content)
        tech_stack = ', '.join([item.strip().strip("'") for item in match_list.group(1).split(',')]) if match_list else None

        # Extract the second string
        match_second_str = re.search(second_str_pattern, doc_content)
        app_desc = match_second_str.group(1).strip() if match_second_str else None

        print(f"App Name: {app_name} App Description: {app_desc} Tech Stack: {tech_stack}") #Debug print
        app_summary = (
                        f"Application Name: {app_name}."
                        f"Application Description: {app_desc}."
                        f"Application Technologies: {tech_stack}."
                        "Application Objectves: Optimal secure implementation."
                      )
        time.sleep(5)
    return app_name, app_summary


#--- PINECONE EMBED ---#


##  Vector embedding and storage
def doc_to_pinecone(docs, index_name, name_space):
    #metadatas = [{"author": "John Doe"}]
    #Save doc in pinecone
    vectorstore_pinecone = PineconeVectorStore.from_documents(
        docs,
        index_name=index_name,
        embedding=bedrock_embeddings,
        namespace=name_space
    )


def text_to_pinecone(texts, metadatas, ids, text_key, index_name, name_space):
    print("ID SENT TO PINECONE")
    print(ids)
    #Save text in pinecone
    vectorstore_pinecone = PineconeVectorStore.from_texts(
        texts,
        embedding=bedrock_embeddings,
        metadatas=metadatas,
        ids=ids,
        text_key=text_key,
        index_name=index_name,
        namespace=name_space
    )


##  Store plain text in pinecone
def plain_text_to_pinecone(userID, plaintext, file_name, id_prefix, file_description):
    texts = [
        plaintext #should contain description, then the plain text content. should it contain name?
    ]

    ########
    if file_description is not None:
        # Metadata for each text
        metadatas = [
            {"page": "1", "source": file_name, "description": file_description, "date": date.today()}
        ]

    else:
        metadatas = [
            {"page": "1", "source": file_name, "date": date.today()}
        ]
    ########

    ########
    if id_prefix is not None:
        # IDs for each record
        ids = id_prefix

    else:
        # IDs for each record might not be so unique
        ids = [file_name]
    ########

    # Other parameters
    #batch_size = 2
    text_key = "text"
    index_name = "rezliant"
    name_space = "customer" + userID
    #upsert_kwargs = {"async_req": True}
    #pool_threads = 4
    #embeddings_chunk_size = 64
    #async_req = True
    #id_prefix = "doc_"

    text_to_pinecone(texts, metadatas, ids, text_key, index_name, name_space)
    success_msg = "Customer ", userID, "information was successfully embedded"
    print("\n", success_msg)
    return success_msg


def customer_biodata_to_pinecone(userID, companyInfo, companySize, position, industry):
    print(userID)
    print(companyInfo)
    print(companySize)
    print(position)
    print(industry)

    biodata = (
                f"Company Info: {companyInfo}."
                f"Company Type: {companySize}."
                f"User Position: {position}."
                f"Industry: {industry}."
                #"These are the details about the user asking the prompts."
              )

    plain_text_to_pinecone(userID, biodata, "Customer Biodata", None, "These are the details about the user asking the prompts")


    return "Successfully Inserted Biodata"


def customer_integrations_to_pinecone(integrations):
    print(integrations)
    return "Intergrations received"


def app_risk_to_pinecone(risk):
    print(risk)
    return "App risk details received"


##  Store customer app details in pinecone - new version
def customer_app_to_pinecone(userID, app_name, app_description, tech_stack):
    
    app_summary = (
                        f"Application Name: {app_name}."
                        f"Application Description: {app_description}."
                        f"Application Technologies: {tech_stack}."
                        "Application Objectves: Optimal secure implementation."
                  )
                      
    
    app_name_stripped = app_name.replace(" ", "") #for vector DB metadata documentation
    print(app_summary)

    texts = [
        app_summary
    ]

    # Metadata for each text
    metadatas = [
        {"page": "1", "source": app_name_stripped, "date": date.today()}
    ]

    # Unique IDs for each text
    ids = [app_name]

    # Other parameters
    #batch_size = 2
    text_key = "text"
    index_name = "rezliant"
    name_space = "customer" + userID
    #upsert_kwargs = {"async_req": True}
    #pool_threads = 4
    #embeddings_chunk_size = 64
    #async_req = True
    #id_prefix = "doc_"

    text_to_pinecone(texts, metadatas, ids, text_key, index_name, name_space)
    success_msg = "Customer ", userID, "application details successfully embedded"
    print("\n", success_msg)
    return success_msg
    
    
##  Store customer app details in pinecone - old version of function
def customer_app_to_pinecone_old(userID):
    documents = mongo_load_app(userID) #user app details from DB
    app_name, app_summary = app_summarize(documents)
    app_name_stripped = app_name.replace(" ", "") #for vector DB metadata documentation
    print(app_summary)

    texts = [
        app_summary
    ]

    # Metadata for each text
    metadatas = [
        {"page": "1", "source": app_name_stripped, "date": date.today()}
    ]

    # Unique IDs for each text
    ids = [app_name]

    # Other parameters
    #batch_size = 2
    text_key = "text"
    index_name = "rezliant"
    name_space = "customer" + userID
    #upsert_kwargs = {"async_req": True}
    #pool_threads = 4
    #embeddings_chunk_size = 64
    #async_req = True
    #id_prefix = "doc_"

    text_to_pinecone(texts, metadatas, ids, text_key, index_name, name_space)
    success_msg = "Customer ", userID, "application details successfully embedded"
    print("\n", success_msg)
    return success_msg


##  Vector store update
def admin_update_vector_store(file_path, index_name, name_space):
    print("Vector store update started...")
    docs = data_ingestion_pinecone_from_docs(file_path)
    #print(docs) #Debug print
    doc_to_pinecone(docs, index_name, name_space)
    print("Vector store update completed.")


def structured_update_vector_store(user_id, file_path, file_name, file_description):
    index_name = "rezliant"
    name_space = "customer" + user_id

    print("Vector store update started...")
    docs = data_ingestion_pinecone_from_docs(file_path, file_description)
    doc_to_pinecone(docs, index_name, name_space)
    print("Vector store update completed.")


def unstructured_update_vector_store(user_id, file_path, file_name, file_description):
    content = read_file(file_path)

    if content is None:
        print("File Can Not Be Read")
        return
    else:
        print("Vector store update started...")
        plain_text_to_pinecone(user_id, content, file_name, None, file_description)
        print("Vector store update completed.")


def stringcontent_update_vector_store(user_id, file_content, file_name, id_prefix, file_description):
    plain_text_to_pinecone(user_id, file_content, file_name, id_prefix, file_description)


def codefile_update_vector_store(user_id, file_content, file_name, id_prefix, file_description):
    """chunks = trees_python.extract_chunks(file_content)  # Get the list of chunks

    for idx, chunk in enumerate(chunks):  # Split chunks into separate items
        print("##########################################################")
        print(chunk)
        print("##########################################################")
        if not chunk.strip():  # Skip empty chunks
            continue
        plain_text_to_pinecone(user_id, chunk, f"{file_name}_chunk{idx+1}", id_prefix, file_description)
        #print("attempt")"""

    chunks = chunker.chunk_source_code(file_content)  # Get the list of chunks

    for idx, chunk in enumerate(chunks):  # Split chunks into separate items
        id_prefix = f"{id_prefix}_chunk{idx+1}"
        id_prefix = [id_prefix]

        print("##########################################################")
        print(chunk)
        print("***************%$%%$%$%********************")
        print(f"{file_name}_chunk{idx+1}")
        print(id_prefix)
        print("##########################################################")
        if not chunk.strip():  # Skip empty chunks
            continue
        plain_text_to_pinecone(user_id, chunk, f"{file_name}_chunk{idx+1}", id_prefix, file_description)



## ========== INSERTION END ========== ##

#-----------------------------------------#

## ========== RETRIEVAL START ========== ##


#--- LLM SELECTION ---#


##RETRIEVE MODEL-ID SUBTYPE
def extract_first_word(text):
    # Use regex to capture the first word before the first hyphen
    match = re.match(r"(\w+)-", text)
    if match:
        return match.group(1)
    return None


##LLM KWARGS DECLARATION
def get_inference_parameters(model_provider, model_subtype): #return a default set of parameters based on the model's provider
    if (model_provider == 'anthropic'): #Anthropic model
        return {
            "max_tokens": 61551,
            "stop_sequences": ["\n\nHuman:"]
        }
    elif (model_provider == 'cohere'): #Cohere
        return {
            "max_tokens": 256,
            "temperature": 0,
            "p": 0.01,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
    elif (model_provider == 'meta'): #Meta
        return {
            "max_gen_len": 256,
            "stop": ["\n\nHuman:", "\nHuman:", "Human:"]
       }
    else: #Amazon
        if (model_subtype == 'titan'):
            return {
                "maxTokenCount": 256,
                "stopSequences": []
            }
        elif (model_subtype == 'nova'):
            return {
                "max_tokens": 512,
                "stopSequences": []
            }
        else:
            return {
                "maxTokenCount": 256,
                "stopSequences": []
            }


def get_llama_llm(model_id):
    model_provider = model_id.split('.')[0] #grab the model provider from the first part of the model id
    model_subtype = extract_first_word(model_id.split('.')[1]) #grab the second part of the model id to know more specifics
    model_kwargs = get_inference_parameters(model_provider, model_subtype)

    llm = ChatBedrock(
        model_id=model_id,
        #model_id="meta.llama3-8b-instruct-v1:0",
        #model_id="cohere.command-r-plus-v1:0",
        #model_id="amazon.titan-text-premier-v1:0",
        #model_id="anthropic.claude-v2:1",
        client=bedrock,
        model_kwargs=model_kwargs,
        region_name=aws_region
    )
    return llm


#--- PINECONE FUNCTIONS ---#


## Pinecone index retrieve
def pinecone_index_retrieve(pinecone, index_name):
    if index_name in pinecone.list_indexes().names():
        index = pinecone.Index(index_name)
        return index


## Pinecone index delete
def pinecone_index_delete(pinecone, index_name):
    if index_name in pinecone.list_indexes().names():
        pinecone.delete_index(index_name)


## Pinecone namespace create
#  Namespaces can only be created through an upsert operation
#  Use text_to_pinecone(texts, metadatas, ids, text_key, index_name, name_space) to create namespace


## Pinecone namespace delete
def pinecone_namespace_delete(pinecone, index, name_space):
    index.delete(namespace=name_space, delete_all=True)


## Pinecone index create
def pinecone_index_create(pinecone, spec, index_name):
    index_exists = False
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)
    index_exists = True
    return index_exists


## Pinecone delete all chunks in a repo
def pinecone_namespace_delete_repo(name_space, id_prefix):
    index = pinecone_instance.Index("rezliant") #target index by name
    #index = pinecone_instance.Index(host="rezliant-m6pbkhr.svc.aped-4627-b74a.pinecone.io'") #target index by host

    for ids in index.list(prefix=id_prefix, namespace=name_space):
        index.delete(ids=ids, namespace=name_space)



#--- CONTEXT DESIGN ---#


## Context Building Algos
def relevance_score(vectorstore_rezliant, documents, query):
    # Step 1: Vectorize the query and documents
    vectorizer = TfidfVectorizer().fit([query] + documents)
    query_vec = vectorizer.transform([query])
    doc_vecs = vectorizer.transform(documents)

    # Step 2: Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vec, doc_vecs).flatten()

    # Debugging cosine similarity scores
    print("\nCosine Similarity Scores:")
    for idx, score in enumerate(cosine_similarities):
        print(f"Document {idx}: {score}")

    max_score_index = cosine_similarities.argmax() #get index of max score
    highest_score_doc = documents[max_score_index] #document on that index
    """highest_score = cosine_similarities[max_score_index] #score on that index
    context = highest_score_doc"""

    for i in range(0, 20):
        highest_score_doc += documents[i]

    context = highest_score_doc

    # Step 3: Filter documents based on similarity threshold
    threshold = 0.1  # Adjust the threshold as needed
    relevant_docs_indices = np.where(cosine_similarities >= threshold)[0]
    relevant_docs = [documents[i] for i in relevant_docs_indices]

    # Print document by document
    """print("\n\nFiltered Results - Relevant Documents (for debugging): ")
    for doc in relevant_docs:
        print("********SEPARATOR*********")
        print(doc)"""

    return context


## Get context and prompt LLM
def get_context_llm_pinecone(vectorstore_rezliant, name_space, query):
    print("building context...")
    try:
        docs_retrieved = vectorstore_rezliant.similarity_search(query, namespace=name_space, k=20)
        # Extract the text content from the document objects
        document_texts = [doc.page_content for doc in docs_retrieved]
        #print(document_texts)

        """i = -1
        for doc in document_texts:
            i += 1
            print("\n\nDocument ", i, ":")
            print(doc)"""

        context = relevance_score(vectorstore_rezliant, document_texts, query)
        #print("===============\nRAG Context: ", context)
        return context
    except Exception as e:
        # If an error occurs, print the traceback for debugging purposes
        traceback.print_exc()

        return "An error occurred while processing the query. Please try again later."


#--- CHAT HISTORY ---#


chat_history_context = []
chat_history_customer = []


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


#--- PROMPT DESIGN ---#


#--- PROMPT ROUTING ---#


######### SYSTEM PROMPT (GENERIC)

system_prompt = (
    "You only answer questions about cybersecurity"
    ##"Answer the human's prompt."
    #"Unless the human's prompt addresses their contextual use-case, incorporate value of context variabe as response" #Basically means customer asks general knowledge question
    #"If human's prompt addresses their contextual use-case, Use the given context to answer the human's prompt." #This means customer asks about how to solve issue for their peculiar context/situation
    #"#You always explain in great detail, you are extremely verbose"
    "You are a very experienced security architect who understands threat modeling and how to build and run threat modeling programmes"
    ##"You are a cyber security  expert who can analyze security vulnerabilities and weaknesses and common vulnerability enumeration (CVE) reports for risk and severity. given each vulnerability’s risk score, you identify the 5 most important issues to address, then provide expert mitigation advice and fixed code, if a code fix is needed."
    "If you don't know the answer, say you don't know."
    ##"Use three sentence maximum and keep the answer concise."
    "Context: {context}"
)
prompt_generic = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

######### SYSTEM PROMPT (MOMENT 1 - CHAT_DEV)

system_prompt_chatdev = (
    "You only answer questions about cybersecurity"
    "You are a secure coding expert and you are speaking to a novice software developer who only wants the required fix so they can copy and paste it"
    "Context: {context}"
)
prompt_chatdev = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_chatdev),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

######### SYSTEM PROMPT (MOMENT 2 - CHAT_DEVSECOPS)

######### SYSTEM PROMPT (MOMENT 3 - CHAT_CISO)

######### SYSTEM PROMPT (MOMENT 4 - CHAT_...)

######### SYSTEM PROMPT (MOMENT 5 - FIX_DEV)

system_prompt_fixdev = (
    "You are a secure coding expert who knows how to fix code and you break down the specific steps to fix my code"
    "Give me a single solution"
    #"Show me the specific commands or instructions that can be directly cut and pasted into an IDE or terminal at the top of the response, followed by additional concise context"
    "Show me specific code block fixes that can be directly cut and pasted into an IDE or terminal at the top of the response, you can give me the explanation afterwards. Also, make sure you explicitly tell me the shortcuts or instructions I can use to navigate my User Interface or IDE anywhere you deem necessary"
    "Focus your instructional steps on one fix pathway at a time, instead of showing multiple approaches to the same solution"
    #"Show the easiest fix first first. If an alternative is asked for, show the next easiest fix. Continue in the order of simpler fixes first whenever asked for an alternative fix"
    "Context: {context}"
)
prompt_fixdev = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_fixdev),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


######### SYSTEM PROMPT (MOMENT 6 - FIX_DEVSECOPS)---(JOSH)
system_prompt_fixdevsecops = (
    "You are an expert secure cloud administrator who knows how to fix security configuration errors and you break down the specific steps to fix my cloud configuration"
    "Give me a single solution"
    "Show me specific configuration fix actions and code that can be directly cut and pasted into a cloud console or terminal at the top of the response, you can give me the explanation afterwards. Also, make sure you explicitly tell me the shortcuts or instructions I can use to navigate my User Interface anywhere you deem necessary"
    "Focus your instructional steps on one fix pathway at a time, instead of showing multiple approaches to the same solution"
    "Context: {context}"
)
prompt_fixdevsecops = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_fixdevsecops),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


######### SYSTEM PROMPT (MOMENT 7 - FIX_CISO)

######### SYSTEM PROMPT (MOMENT 8 - FIX_...)

######### SYSTEM PROMPT (MOMENT 9 - SCAN_RANKER)

system_prompt_scanranker = (
    "You are a secure coding expert who specializes in identifying vulnerabilities and ranking them according to how dangerous they can be"
    "Always respond with a ranked list of IDs only. For example: [2,1,3]"
    "Context: {context}"
)
prompt_scanranker = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_scanranker),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
##JOSH
system_prompt_cloudscanranker = (
    "You are an expert secure cloud administrator who specializes in identifying vulnerabilities and ranking them according to how dangerous they can be"
    "Always respond with a ranked list of IDs only. For example: [2,1,3]"
    "Context: {context}"
)
prompt_cloudscanranker = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_cloudscanranker),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
#--- END PROMPT ROUTING ---#


def get_context_response_llm_pinecone(llm, vectorstore_rezliant, query, persona, moment): #sends vector namespace to llm directly
    try:
        start_time = time.time()
        # Create history-aware retriever for rezliant namespace
        history_aware_retriever = create_history_aware_retriever(llm, vectorstore_rezliant.as_retriever(), contextualize_q_prompt)
        prompt = ""

        
        # Select the system prompt
        if moment == "chat" and persona == "software developer":
            print("Chatdev")
            prompt = prompt_chatdev
        elif moment == "chat" and persona == "devsecops":
            prompt = prompt_chatdevsec
        elif moment == "chat" and persona == "devops":
            prompt = prompt_chatdevops
        elif moment == "chat" and persona == "ceo":
            prompt = prompt_chatceo
        elif moment == "fix" and persona == "software developer":
            prompt = prompt_fixdev
        elif moment == "fix" and persona == "devsecops":
            prompt = prompt_fixdevsecops  # Updated here
        elif moment == "fix" and persona == "devops":
            prompt = prompt_fixdevops
        elif moment == "fix" and persona == "ceo":
            prompt = prompt_fixceo
        elif moment == "scan" and persona == "ranker":
            prompt = prompt_scanranker
        elif moment == "cloudscan" and persona == "ranker":  # Added for cloud scan ranking(Josh)
            prompt = prompt_cloudscanranker
        else:
            prompt = prompt_generic

        # Create the question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # Create the retrieval chain
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Invoke the retrieval chain with the query
        response = chain.invoke({"input": query, "chat_history": chat_history_context})
        chat_history_context.extend([HumanMessage(content=query), AIMessage(content=response["answer"])])
        print("===============", "\nRAG Context:\n")
        #print("Prompt: ", prompt) #Debug print
        #print(response["input"])
        #print(response["context"])
        print(response["answer"])
        #print("===============", "\nContext Chat History: \n")
        #print(chat_history_context)
        end_time = time.time()
        elapsed_time_for_context = round(end_time - start_time, 2)
        return response["answer"], elapsed_time_for_context
    except Exception as e:
        # If an error occurs, print the traceback for debugging purposes
        traceback.print_exc()
        return "An error occurred while processing the query. Please try again later."



def get_response_llm_pinecone(llm, vectorstore_customer, query, persona, moment):
    try:
        start_time = time.time()
        # Create history-aware retriever for customer namespace
        history_aware_retriever = create_history_aware_retriever(llm, vectorstore_customer.as_retriever(), contextualize_q_prompt)
        prompt = ""

        # Select the system prompt
        # Select the system prompt
        if moment == "chat" and persona == "software developer":
            print("Chatdev")
            prompt = prompt_chatdev
        elif moment == "chat" and persona == "devsecops":
            prompt = prompt_chatdevsec
        elif moment == "chat" and persona == "devops":
            prompt = prompt_chatdevops
        elif moment == "chat" and persona == "ceo":
            prompt = prompt_chatceo
        elif moment == "fix" and persona == "software developer":
            prompt = prompt_fixdev
        elif moment == "fix" and persona == "devsecops":
            prompt = prompt_fixdevsecops  # Updated here
        elif moment == "fix" and persona == "devops":
            prompt = prompt_fixdevops
        elif moment == "fix" and persona == "ceo":
            prompt = prompt_fixceo
        elif moment == "scan" and persona == "ranker":
            prompt = prompt_scanranker
        elif moment == "cloudscan" and persona == "ranker":  # Added for cloud scan ranking(Josh)
            prompt = prompt_cloudscanranker
        else:
            prompt = prompt_generic

        # Create the question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # Create the retrieval chain
        #chain = create_retrieval_chain(vectorstore_customer.as_retriever(), question_answer_chain)
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Invoke the retrieval chain with the query
        #response = chain.invoke({"input": query, "context": context, "chat_history": chat_history_customer})
        response = chain.invoke({"input": query, "chat_history": chat_history_customer})
        chat_history_customer.extend([HumanMessage(content=query), AIMessage(content=response["answer"])])
        print("===============","\nRAG Final Response:\n")
        #print("Prompt: ", prompt) #Debug print
        #print(response["input"]_
        #print(response["context"])
        print(response["answer"])
        #print("===============","\nCustomer Chat History: \n")
        #print(chat_history_customer)
        #combined_output = "Generally,\n\n" + context + "\n\nSpecifically,\n\n" + response["answer"]
        #return combined_output
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        return response["answer"] + "\nTime: " + str(elapsed_time) + "s for LLM to create Final Response"
    except Exception as e:
        # If an error occurs, print the traceback for debugging purposes
        traceback.print_exc()
        return "An error occurred while processing the query. Please try again later."



## ========== RETRIEVAL END ========== ##



## ========== RATE LIMIT START ========== ##


#--- EXPONENTIAL BACKOFF RETRY ---#

def exponential_backoff_retry(func, max_retries=5, base_delay=1, max_delay=60):
    for attempt in range(max_retries):
        try:
            # Attempt to call the provided function
            return func()
        except botocore.exceptions.ThrottlingException:
            delay = min(max_delay, base_delay * (2 ** attempt))  # Exponential backoff
            delay += random.uniform(0, 1)  # Add jitter to avoid retry bursts
            print(f"Throttling detected. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        except Exception as e:  # Catch other errors normally
            raise e  # Raise if it's a different exception
    print(f"Failed after {max_retries} retries")
    return None

## ========== RATE LIMIT END ========== ##




#====================MAIN==========================
#==================================================


#--- CHAT API ROUTE ---#
def response_to_prompt(moment, user_id, persona, prompt, llm_model_id):

    def make_llm_call():
        print("\n*****====*****\n" + prompt + "\n")
        index_name = "rezliant" #name of index in pinecone database

        # Connect to pinecone
        index = pinecone_index_retrieve(pinecone_instance, index_name)

        #if index is None:
            #No such index exists
            #index = pinecone_index_create(pinecone_instance, spec_instance, index_name)
        #else:
            #Index exists, embed data
            #file_path = "data/story0.pdf"
            #update_vector_store(file_path, index_name) #embed_data
           #print(index.describe_index_stats()) #structure

        # Retrieve embeddings from Pinecone vector store index
        text_field = "text"
        name_space_rezliant = "rezdocuments"
        #name_space_customer1 = "customer94b8a448-00a1-704b-f8d8-3452bbb61092"
        name_space_customer1 = "customer" + user_id #"customer94b8a448-00a1-704b-f8d8-3452bbb61092"

        vectorstore_rezliant = PineconeVectorStore(index_name=index_name, embedding=bedrock_embeddings, namespace=name_space_rezliant)
        vectorstore_customer = PineconeVectorStore(index, bedrock_embeddings, text_field, name_space_customer1)

        # Send embeddings to LLM
        llm = get_llama_llm(llm_model_id)
        #context_response_llm, elapsed_time_for_context = get_context_response_llm_pinecone(llm, vectorstore_rezliant, prompt, persona, moment)
        response = get_response_llm_pinecone(llm, vectorstore_customer, prompt, persona, moment)
        #combined_output = "Generally,\n\n" + context_response_llm + "\n\nSpecifically,\n\n" + response
        #return combined_output

        return response

    # Wrap the LLM call with retry and backoff logic
    return exponential_backoff_retry(make_llm_call)


#--- FIX API ROUTE ---#
def response_with_fix(moment, user_id, persona, prompt, llm_model_id):

    def make_llm_call():
        print("\n*****====*****\n" + prompt + "\n")
        index_name = "rezliant" #name of index in pinecone database

        # Connect to pinecone
        index = pinecone_index_retrieve(pinecone_instance, index_name)

        #if index is None:
            #No such index exists
            #index = pinecone_index_create(pinecone_instance, spec_instance, index_name)
        #else:
            #Index exists, embed data
            #file_path = "data/story0.pdf"
            #update_vector_store(file_path, index_name) #embed_data
           #print(index.describe_index_stats()) #structure

        # Retrieve embeddings from Pinecone vector store index
        text_field = "text"
        name_space_rezliant = "rezdocuments"
        name_space_customer1 = "customer" + user_id #"customer94b8a448-00a1-704b-f8d8-3452bbb61092"

        vectorstore_rezliant = PineconeVectorStore(index_name=index_name, embedding=bedrock_embeddings, namespace=name_space_rezliant)
        vectorstore_customer = PineconeVectorStore(index, bedrock_embeddings, text_field, name_space_customer1)

        # Send embeddings to LLM
        llm = get_llama_llm(llm_model_id)
        #context_response_llm, elapsed_time_for_context = get_context_response_llm_pinecone(llm, vectorstore_rezliant, prompt, persona, moment)
        response = get_response_llm_pinecone(llm, vectorstore_customer, prompt, persona, moment)

        return response

    # Wrap the LLM call with retry and backoff logic
    return exponential_backoff_retry(make_llm_call)


## User creates app -
#  pass me userid only after app details were succesfully inserted in db from frontend
#userID = "104570994339428114938" #Food Ordering
"""userID = "116270371301858262667" #Hospital Healthcare
customer_app_to_pinecone(userID)"""

## Test Query
#question = "What’s the best security certification to focus on as a healthcare/finance/specific industry company?"
#question = "Please describe general methods for assigning risk to weaknesses uncovered through a threat model"
"""question = "What are the relationships between secure design, security architecture, and threat modeling, if any?"
question1 = "How is risk used during threat modeling?"
question2 = "What function does risk play in a threat model?"
question3 = "hello"""
#question4 = "hello, how are you?"
#print("*******/////\\\\\\\\*******\n\n")
#response_to_prompt(question, "anthropic.claude-3-sonnet-20240229-v1:0")
"""response_to_prompt(question, "anthropic.claude-3-5-sonnet-20240620-v1:0")
response_to_prompt(question1, "anthropic.claude-3-5-sonnet-20240620-v1:0")
response_to_prompt(question2, "anthropic.claude-3-5-sonnet-20240620-v1:0")
response_to_prompt(question3, "anthropic.claude-3-5-sonnet-20240620-v1:0")"""
#response_to_prompt("chat", "94b8a448-00a1-704b-f8d8-3452bbb61092", "software developer", question4, "amazon.nova-pro-v1:0")

#amazon.nova-micro-v1:0

## Delete Namespace
"""index_name = "rezliant"
name_space = "abc"
index = pinecone_index_retrieve(pinecone_instance, index_name)
print(index.describe_index_stats())
pinecone_namespace_delete(pinecone_instance, index, name_space)
print(index.describe_index_stats())"""

#print(pinecone_instance.describe_index("rezliant"))

"""index = pinecone_instance.Index("rezliant") #target index by name
#index = pinecone_instance.Index(host="rezliant-m6pbkhr.svc.aped-4627-b74a.pinecone.io'") #target index by host

for ids in index.list(prefix='r2lomo/juice-shop#', namespace='customerc43864d8-5081-7011-74ce-2f9fcb565460'):
    index.delete(ids=ids, namespace='customerc43864d8-5081-7011-74ce-2f9fcb565460')
    #print(ids)"""
