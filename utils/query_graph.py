from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .card_filters import CardFilters, QUERY_PARSER_CHAIN
from .logging_config import LOGGER


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="credit_cards",
    embedding=embeddings,
)


class GraphState(TypedDict):
    user_input: str
    query_intent: CardFilters | None
    context: List[Document]


def _metadata_filter(query_intent):
    metadata_filter = {"must": [], "should": []}

    renewal_fee = query_intent["acceptable_fees"]
    domestic_lounge_access = query_intent["has_domestic_lounge_access"]
    golf_benefits = query_intent["has_golf_benefits"]
    international_lounge_access = query_intent["has_international_lounge_access"]
    movie_benefits = query_intent["has_movie_benefits"]
    travel_benefits = query_intent["has_travel_benefits"]

    if renewal_fee:
       metadata_filter["must"].append({"key": "metadata.renewal_fee", "range": {"lte": renewal_fee}})

    if domestic_lounge_access:
        metadata_filter["should"].append({"key": "metadata.has_domestic_lounge_access", "match": {"value": True}})

    if golf_benefits:
        metadata_filter["should"].append({"key": "metadata.has_golf_benefits", "match": {"value": True}})

    if movie_benefits:
        metadata_filter["should"].append({"key": "metadata.has_movie_benefits", "match": {"value": True}})

    if international_lounge_access:
        metadata_filter["should"].append({"key": "metadata.has_international_lounge_access", "match": {"value": True}})

    if travel_benefits:
        metadata_filter["should"].append({"key": "metadata.has_travel_benefits", "match": {"value": True}})

    print(metadata_filter)
    return metadata_filter


def extract_query_intent_node(state: GraphState) -> GraphState:
    """Use the query parser chain to extract intent from the user input"""
    LOGGER.info("Parsing User Query to extract intent...")
    query_intent = QUERY_PARSER_CHAIN.invoke({"user_input": state["user_input"]})
    state["query_intent"] = query_intent
    return state


def retrieve_documents_node(state: GraphState) -> GraphState:
    """Generate Metadata filters using extracted query intent"""
    LOGGER.info("Generating metadata filters...")
    query = state["user_input"]
    query_intent = state["query_intent"]
    metadata_filter = _metadata_filter(query_intent.model_dump())
    results = vector_store.similarity_search(query=query,
                                             k=10,
                                             filter=metadata_filter)
    state["context"] = results
    return state


workflow = StateGraph(GraphState)

workflow.add_node("extract_query_intent", extract_query_intent_node)
workflow.add_node("retrieve_documents", retrieve_documents_node)

workflow.add_edge(START, "extract_query_intent")
workflow.add_edge("extract_query_intent", "retrieve_documents")
workflow.add_edge("retrieve_documents", END)

FEATURE_EXTRACTOR_GRAPH = workflow.compile()