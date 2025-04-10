from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class CardFilters(BaseModel):

    acceptable_fees: int | None = Field(...,
        description="The maximum annual fee the user "
        "is willing to pay."
    )
    has_movie_benefits: bool | None = Field(...,
        description="Does the user want Movie Benefits"
    )
    has_dining_benefits: bool | None = Field(...,
        description="Does the user want Dining Benefits"
    )
    has_travel_benefits: bool | None = Field(...,
        description="Does the user want Travel Benefits"
    )
    has_domestic_lounge_access: bool | None = Field(...,
        description="Does the user want Domestic "
        "Lounge Access Benefits"
    )
    has_international_lounge_access: bool | None = Field(...,
        description="Does the user want International "
        "Lounge Access Benefits"
    )
    has_golf_benefits: bool | None = Field(...,
        description="Does the user want Golf Benefits"
    )
    top_spend_categories: List[str] | None = Field(...,
        description="The user's top spending categories.",
    )
    top_merchant_brands: List[str] | None = Field(...,
        description="The top merchant brands where the user "
        "spends the most money."
    )


query_parser_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""
You are an AI assistant specializing in extracting query intent for credit card recommendations. Your task is to analyze user input and extract key requirements in JSON format.

**Instructions:**

1.  **Analyze the input text** to identify the information requested below.
2.  **Output the information as a JSON object** with the following keys:

    *   `acceptable_fees`: (int) The maximum acceptable annual fee the user is willing to pay for a credit card. If the user wants a lifetime free card, set the acceptable fees to 0. If the user mentions a specific fee they are willing to pay (e.g., "up to rs 5000"), extract that number. If they mention a range (e.g., "between rs 1000 and 5000"), extract the *maximum* value of the range (e.g., 5000). If no fee is mentioned, or the user indicates a willingness to consider cards with fees but provides no specific limit, leave it as `null`.)
    *   `has_movie_benefits`: (bool) Whether the user wants any movie-related benefits. If no movie benefits requirements are found, set to `False`.
    *   `has_dining_benefits`: (bool) Whether the user wants any dining-related benefits. If no dining benefits requirements are found, set to `False`.
    *   `has_travel_benefits`: (bool) Whether the user wants any travel-related benefits (excluding lounge access). If no travel benefits requirements are found, set to `False`.
    *   `has_domestic_lounge_access`: (bool) Whether the user wants complimentary domestic lounge access. If none, set to `False`.
    *   `has_international_lounge_access`: (bool) Whether the user wants complimentary international lounge access. If none, set to `False`.
    *   `has_golf_benefits`: (bool) Whether the user wants any golf-related benefits. If no golf benefits requirements are found, set to `False`.
    *   `has_insurance_benefits`: (bool) Whether the user wants any insurance benefits (e.g., travel insurance, purchase protection). If no insurance benefits requirements are found, set to `False`.
    *   `top_merchant_brands`: (list of strings) A list of the top merchant brands where the user spends the most money. e.g., ["Swiggy", "Amazon", "Flipkart"]
    *   `top_spend_categories`: (list of strings) A list of the top spending categories based on the user's spending habits. e.g., ["Shopping", "Travel", "Dining", "Lifestyle"]. If you are not able to extract the top spend categories, identify the most relevant categories to the top merchants.

    
3.  **Be precise and avoid making assumptions.** Only include information explicitly stated in the provided text.
4.  **If a specific detail is not mentioned in the text,** set its corresponding value to `False` where applicable.
5.  For fees, do not include currency symbol if available, just provide the numerical value.
"""
        ),
        (
            "user",
            "User Input: {user_input}"
        )
    ]
)


rate_limiter = InMemoryRateLimiter(requests_per_second=0.2,
                                   check_every_n_seconds=0.1,
                                   max_bucket_size=10)
query_parser_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",
                                                 rate_limiter=rate_limiter)
QUERY_PARSER_CHAIN = (query_parser_prompt | 
                      query_parser_model.with_structured_output(CardFilters))
