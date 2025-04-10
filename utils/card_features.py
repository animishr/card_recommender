from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class MerchantOffers(BaseModel):
    merchant_brand: str | None = Field(...,
        description="Merchant Brand Name"
    )
    offers: List[str] | None = Field(...,
        description="Offers on spends on the merchant "
        "brand"
    )


class CardFeatures(BaseModel):

    product_name: str | None = Field(...,
        description="Credit Card Product Name"
    )
    joining_fee: int | None = Field(...,
        description="Joining Fee"
    )
    renewal_fee: int | None = Field(...,
        description="Renewal Fee"
    )
    best_suited_categories: List[str] | None = Field(...,
        description="Best suited Categories"
    )
    rewards_type: List[str] | None = Field(...,
        description="Reward Types"
    )
    welcome_benefits: str | None = Field(
        default=None,
        description="Welcome Benefits (if any)"
    )
    movie_benefits: str | None = Field(
        default=None,
        description="Movie Benefits (if any)"
    )
    dining_benefits: str | None = Field(
        default=None,
        description="Dining Benefits (if any)"
    )
    travel_benefits: str | None = Field(
        default=None,
        description="Travel Benefits (if any)"
    )
    domestic_lounge_access_annual: int | None = Field(...,
        description="Number of Domestic Lounge Visits "
        "allowed annually"
    )
    domestic_lounge_access_quarterly: int | None = Field(...,
        description="Number of Domestic Lounge Visits "
        "allowed quarterly"
    )
    international_lounge_access_annual: int | None = Field(...,
        description="Number of International Lounge Visits "
        "allowed annually"
    )
    international_lounge_access_quarterly: int | None = Field(...,
        description="Number of International Lounge Visits "
        "allowed quarterly"
    )
    golf_benefits: str | None = Field(
        default=None,
        description="Golf Benefits (if any)"
    )
    insurance_benefits: str | None = Field(
        default=None,
        description="Insurance Benefits (if any)"
    )
    waiver_amount: int | None = Field(...,
        description="Renewal Fee waiver amount"
    )
    foreign_currency_markup: float | None = Field(...,
        description="Foreign Currency Markup in percentage"
    )
    merchant_offers: List[MerchantOffers] | None = Field(...,
        description="All the merchant offers"
    )


profiler_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""
You are an AI assistant specialized in parsing and extracting information from credit card product descriptions. Your objective is to analyze the provided text describing a credit card product and extract specific details accurately.

# Task: Analyze Text and Output JSON

Analyze the credit card description text provided below the "---" separator. Extract the requested information and format it **strictly** as a single JSON object.

# Output Format: JSON Object

Do not include any introductory text, explanations, or markdown formatting around the JSON. The JSON object must have the following keys with the specified data types and constraints:

*   `product_name`: (String) The official name of the credit card product as mentioned in the text.
*   `joining_fee`: (Integer or `null`) The one-time joining fee. Extract only the numerical value (e.g., `500`). If not mentioned, use `null`.
*   `renewal_fee`: (Integer or `null`) The annual renewal fee. Extract only the numerical value (e.g., `500`). If not mentioned, use `null`.
*   `best_suited_categories`: (List of Strings) A list identifying the primary spending categories where the card offers the most significant benefits (e.g., `["Travel", "Dining", "Shopping"]`). Base this on emphasized reward rates or specific perks. If no specific categories are clearly highlighted as 'best', infer from the strongest benefit areas mentioned, or use an empty list `[]` if unclear.
*   `card_rewards_type`: (List of Strings) The primary type(s) of rewards offered (e.g., `["Cashback"]`, `["Reward Points"]`, `["Airmiles"]`). Use an empty list `[]` if not mentioned.
*   `welcome_benefits`: (String or `null`) A concise description of the welcome benefits offered to new cardholders. If none are explicitly mentioned, use `null`.
*   `movie_benefits`: (String or `null`) A concise description of any movie-related benefits (e.g., ticket discounts, BOGO offers). If none are mentioned, use `null`.
*   `dining_benefits`: (String or `null`) A concise description of any dining-related benefits (e.g., discounts, specific programs). If none are mentioned, use `null`.
*   `travel_benefits`: (String or `null`) A concise description of any travel-related benefits, **excluding** specific lounge access counts (e.g., discounts on bookings, reward points on travel spend, travel credits). If none are mentioned, use `null`.
*   `domestic_lounge_access_annual`: (Integer or `null`) The total number of complimentary domestic airport lounge accesses per year. If only quarterly access is mentioned, calculate the annual total if possible (e.g., 2 per quarter = 8 per year); otherwise, if only quarterly is stated without enough info for annual calculation or if not mentioned, use `null`.
*   `domestic_lounge_access_quarterly`: (Integer or `null`) The number of complimentary domestic airport lounge accesses per quarter. If only annual access is mentioned, or if not mentioned at all, use `null`.
*   `international_lounge_access_annual`: (Integer or `null`) The total number of complimentary international airport lounge accesses per year. Similar calculation logic applies as for domestic annual access. If not mentioned, use `null`.
*   `international_lounge_access_quarterly`: (Integer or `null`) The number of complimentary international airport lounge accesses per quarter. Similar logic applies as for domestic quarterly access. If not mentioned, use `null`.
*   `golf_benefits`: (String or `null`) A concise description of any golf-related benefits (e.g., complimentary games, lesson discounts). If none are mentioned, use `null`.
*   `insurance_benefits`: (String or `null`) A concise description of any insurance benefits provided (e.g., travel insurance, air accident cover, purchase protection). If none are mentioned, use `null`.
*   `renewal_fee_waiver_spend`: (Integer or `null`) The annual spending amount required to waive the renewal fee. Extract only the numerical value. If not mentioned, use `null`.
*   `foreign_currency_markup`: (Float or `null`) The percentage charged as foreign currency markup. Extract only the numerical value (e.g., `3.5`). Do not include the "%" symbol. If not mentioned, use `null`.
*   `merchant_offers`: (List of Objects) A list where each string summarizes a specific merchant offer or discount available with the card (e.g., `["15% off at Myntra", "10% discount on Swiggy orders over 500"]`). If no specific merchant offers are listed, use an empty list `[]`.

# Rules & Constraints:

1.  **Strict Extraction:** Only include information explicitly stated in the provided text. Do not make assumptions or infer details not present.
2.  **Precision:** Be precise with names, numbers, and descriptions.
3.  **Handle Missing Data:** If a specific detail is not mentioned, use `null` for string/integer/float fields or an empty list `[]` for list fields as specified above.
4.  **Numeric Values:** For fees, spend waivers, and markup, extract only the numerical value. Do not include currency symbols (e.g., $, ₹, £) or percentage signs (%).
5.  **Lounge Access Logic:** Carefully check if lounge access is specified as annual or quarterly, domestic or international. Calculate annual from quarterly if explicitly stated (e.g., "2 per quarter"). If only a total number is given without specifying frequency, assume annual unless context strongly suggests otherwise. If ambiguous, prioritize `null`.
"""
        ),
        (
            "user",
            "Product Information: {product_information}"
        )
    ]
)


rate_limiter = InMemoryRateLimiter(requests_per_second=0.2,
                                   check_every_n_seconds=0.1,
                                   max_bucket_size=10)
feature_extractor_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",
                                                 rate_limiter=rate_limiter)
FEATURE_EXTRACTOR_CHAIN = (profiler_prompt | 
                           feature_extractor_model.with_structured_output(CardFeatures))
