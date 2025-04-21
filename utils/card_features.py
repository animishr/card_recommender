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
You are a highly skilled credit card information extraction expert. Your task is to analyze credit card product summaries and extract key details into a structured format. Your goal is to capture ALL possible merchant brand offers, even if they seem minor.

**Input:**

You will be provided with a text-based credit card product summary containing the following information:

*   Product Name:
*   Product Description:
*   Product Details: (Bullet points)
*   Joining Fee:
*   Renewal Fee:
*   Best Suited For:
*   Reward Type:
*   Welcome Benefits:
*   Movie & Dining:
*   Travel:
*   Domestic Lounge Access:
*   International Lounge Access:
*   Golf:
*   Insurance Benefits:
*   Spend-Based Waiver:
*   Foreign Currency Markup:
*   Interest Rates:
*   Fuel Surcharge:

**Task:**

Carefully read the provided credit card product summary and extract the following information. Ensure the extracted information is accurate, concise, and well-formatted.
** product_name **: The official name of the credit card product as mentioned in the text.
** joining_fee **: The one-time joining fee (numerical value, e.g., '499' or '0').  If a range is given, provide the lower bound.
** renewal_fee **: The annual renewal fee (numerical value, e.g., '999' or '0'). If a range is given, provide the lower bound.
** best_suited_categories **: A list identifying the primary spending categories where the card offers the most significant benefits (e.g., ['Travel', 'Dining', 'Shopping', 'Entertainment', 'Groceries', 'Fuel']). If 'Best Suited For' is explicitly mentioned, use that information. Otherwise, infer the categories based on the 'merchant_offers'. If no merchant offers are found, or they are too diverse to infer a category, use 'General Spending'. Examples: ['Travel', 'Dining'], ['Shopping', 'Entertainment'], ['Groceries', 'Fuel'], ['General Spending'].
** card_rewards_type **: A list of the primary type(s) of rewards offered (e.g., ['Cashback'], ['Reward Points'], ['Airmiles']).
** welcome_benefits **: A concise description of the welcome benefits offered to new cardholders.
** movie_benefits **: A concise description of any movie-related benefits (e.g., ticket discounts, BOGO offers).
** dining_benefits **: A concise description of any dining-related benefits (e.g., discounts, specific programs).
** travel_benefits **: A concise description of any travel-related benefits, excluding specific lounge access counts (e.g., discounts on bookings, reward points on travel spend, travel credits).
** domestic_lounge_access_annual **: The total number of complimentary domestic airport lounge accesses per year (numerical value, e.g., '8' or '0'). If access is unlimited, specify "1000".
** domestic_lounge_access_quarterly **: The number of complimentary domestic airport lounge accesses per quarter (numerical value, e.g., '2' or '0'). If access is unlimited, specify "1000".
** international_lounge_access_annual **: The total number of complimentary international airport lounge accesses per year (numerical value, e.g., '4' or '0'). If access is unlimited, specify "1000".
** international_lounge_access_quarterly **: The number of complimentary international airport lounge accesses per quarter (numerical value, e.g., '1' or '0'). If access is unlimited, specify "1000".
** golf_benefits **: A concise description of any golf-related benefits (e.g., complimentary games, lesson discounts).
** insurance_benefits **: A concise description of any insurance benefits provided (e.g., travel insurance, air accident cover, purchase protection).
** renewal_fee_waiver_spend **: The annual spending amount required to waive the renewal fee (numerical value, e.g., '150000' or '0' if no waiver is available).
** foreign_currency_markup **: The percentage charged as foreign currency markup (numerical value, e.g., '3.5').
** merchant_offers **: A list of dictionaries, where each dictionary represents a merchant brand and its list of offers. Include ALL offers or discounts available with the card, including seemingly minor offers AND any reward points or loyalty programs SPECIFICALLY offered by or related to the merchant. If NO specific merchant offers are found, the value should be `null`

**Instructions:**

* Pay close attention to numerical values for fees, lounge access, and spending requirements. Extract these accurately.
* Be concise and avoid including extraneous information in the descriptions.
* For merchant_offers, extract ALL offers with specific, named merchants, even if the offer seems small or insignificant. This INCLUDES any reward points multipliers or loyalty program memberships offered specifically by that merchant. If no merchant offers are found, the value should be `null`.
* For best_suited_categories, first check if the information is explicitly provided under a heading like 'Best Suited For'. If it is, use that information. If not, or if the explicitly provided categories are too general, use the merchant_offers to infer the most appropriate categories. For example, if the merchant_offers include Myntra, Amazon, and Flipkart, then best_suited_categories should be ['Shopping']. If they include Swiggy and Zomato, then best_suited_categories should be ['Dining']. If the offers are too diverse (e.g., one travel offer, one dining offer, one shopping offer), or if there are no merchant offers, then set best_suited_categories to ['General Spending'].
* For movie_benefits and dining_benefits, carefully extract the information and SEPARATE the benefits even if they are presented together in a combined 'Movie & Dining' section. Consider movie-related benefits to include ticket discounts, BOGO offers, and access to premium cinema experiences. Consider dining-related benefits to include discounts at restaurants, access to dining programs, and complimentary meals. If a specific benefit type is not mentioned, set the corresponding value to `null`.
* For travel_benefits, extract the information directly from the corresponding sections in the input. If no travel benefits are found, set the value to `null`. Exclude lounge access benefits.
* Prioritize information directly stated in the product summary. Do not make assumptions or inferences.
* Ensure all lists are properly formatted and contain relevant values only.
* If a specific piece of information is not explicitly mentioned, set the corresponding value to `null`.
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
