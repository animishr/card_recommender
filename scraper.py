import requests

from uuid import uuid4

from time import sleep
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from utils.credit_card import CreditCard
from utils.card_parser import FEATURE_EXTRACTOR_GRAPH
from utils.logging_config import LOGGER


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="credit_cards",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="credit_cards",
    embedding=embeddings,
)

response = requests.get('https://cardinsider.com/card-issuer/')
soup = BeautifulSoup(response.content, 'html.parser')
issuers = soup.find_all('div', class_='item-new')
credit_cards = []

for issuer in issuers[:1]:
    a_ = issuer.find('a')
    bank_name = a_.text
    bank_url = f"https://cardinsider.com{a_['href']}"
    
    r = requests.get(bank_url)
    sp = BeautifulSoup(r.content, 'html.parser')
    cards = sp.find_all('div', class_='single_credit_card_box')
    docs = []

    for i, card in enumerate(cards[:30], 1):
        a_ = card.find('a', class_='title_list_link')
        card_name = a_.text
        card_url = a_["href"]
        
        LOGGER.info("Iteration %d: Extracting Data for '%s'", i, card_name)
        cc = CreditCard(card_url)
        card_summary = cc.to_text()

        try:
            initial_state = {"product_information": card_summary}
            final_state = FEATURE_EXTRACTOR_GRAPH.invoke(initial_state)

            card_features = final_state["card_features"].model_dump()
            document = Document(page_content=card_summary, metadata=card_features)
            
            docs.append(document)
        
        except:
            LOGGER.info("Failed extracting Data for '%s'. Skipping...", card_name)
            sleep(10)


uuids = [str(uuid4()) for _ in range(len(docs))]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)
vector_store.add_documents(documents=docs, ids=uuids)
