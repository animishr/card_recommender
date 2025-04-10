from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from utils.logging_config import LOGGER
from utils.card_features import FEATURE_EXTRACTOR_CHAIN, CardFeatures


class GraphState(TypedDict):
    product_information: str
    card_features: CardFeatures | None


def parse_card_features_node(state: GraphState) -> GraphState:
    """Use the profiler chain to extract fields from the user input"""
    LOGGER.info("Parsing Product Information...")
    card_features = FEATURE_EXTRACTOR_CHAIN.invoke({"product_information": state["product_information"]})
    state["card_features"] = card_features
    return state


workflow = StateGraph(GraphState)

workflow.add_node("parse_card_features", parse_card_features_node)
workflow.add_edge(START, "parse_card_features")
workflow.add_edge("parse_card_features", END)

FEATURE_EXTRACTOR_GRAPH = workflow.compile()
