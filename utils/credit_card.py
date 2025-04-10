import logging
import re

import requests

from bs4 import BeautifulSoup

from .utils import json_to_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEE_PATTERN = re.compile(r'(?P<currency>Rs\.?|\u20b9) ?(?P<amount>[\d,]+) \+ GST')

class CreditCard:

    def __init__(self, url):
        logger.info("Fetching Data for URL: %s", url)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.product_name = self._get_product_name(soup)
        self.product_description = self._get_product_description(soup)
        self.summary = self._get_summary(soup)
        self.rewards = self._get_rewards(soup)
        self.fees_charges = self._get_fees_charges(soup)
        self.product_details = self._get_product_details(soup)
        self.pros_cons = self._get_pros_cons(soup)


    @staticmethod
    def _get_product_name(soup):
        return soup.find('h1', class_='card-main-title').text.strip()
    

    @staticmethod
    def _get_product_description(soup):
        content =  soup.find('div', class_='content-top-section')
        paragraphs = content.find_all('p')
        return ' '.join(p.text.strip() for p in paragraphs)


    @staticmethod
    def _get_clean_fees(fee_string):
        if fee_string.lower() == "nil":
            return 0
        
        m = re.match(FEE_PATTERN, fee_string)
        if m:
            return int(re.sub(',', '', m.groupdict()["amount"]))
        else:
            return fee_string
        

    @staticmethod
    def _get_categories_rewards(value):
        return [v for v in value.split("|") if v.strip()]


    def _get_summary(self, soup):
        summary = soup.find('div', class_='fees-subpart')
        keys = summary.find_all('h4', class_='list_credit_title')
        values = summary.find_all('div', class_='col-md-9')

        dict_ = {}
        for k, v in zip(keys, values):
            key = k.text.strip()
            value = v.text.strip()

            if key.endswith("Fee"):
                dict_[key] = self._get_clean_fees(value)

            elif key in {"Best Suited For", "Reward Type"}:
                dict_[key] = self._get_categories_rewards(value)

            elif value.lower() in {'na', 'n/a'}:
                dict_[key] = None

            else:
                dict_[key] = value

        return dict_


    @staticmethod
    def _get_rewards(soup):
        rewards = soup.find('div', attrs={"id": "rewards-and-benefits"})
        keys = rewards.find_all('h4')
        values = rewards.find_all('p')

        dict_ = {}
        for k, v in zip(keys, values):
            key = k.text.strip()
            value = v.text.strip()

            if value.lower() in {'na', 'n/a'}:
                dict_[key] = None
            else:
                dict_[key] = value

        return dict_


    @staticmethod
    def _get_fees_charges(soup):
        fees_charges = soup.find('div', attrs={"id": "Fees-Charges"})
        keys = fees_charges.find_all('h4')
        values = fees_charges.find_all('p')

        return dict((key.text.strip(), value.text.strip()) for key, value in zip(keys, values))

    @staticmethod
    def _get_product_details(soup):
        product_details = soup.find('div', attrs={"id": "Product-Details"})
        details = product_details.find_all('li')

        return [detail.text.strip() for detail in details]

    @staticmethod
    def _get_pros_cons(soup):
        pros_cons = soup.find('div', attrs={"id": "Pros-Cons"})
        if pros_cons:
            pros = pros_cons.find('div', class_="Pros-sec").find_all('li')
            cons = pros_cons.find('div', class_="Cons-sec").find_all('li')

            return {
                        "pros": [pro.text for pro in pros] if pros else None,
                        "cons": [con.text for con in cons] if cons else None
                    }
        else:
            return {"pros": None, "cons": None}
        

    def to_dict(self):
        dict_ = {
                    "Product Name": self.product_name,
                    "Product Description": self.product_description,
                    "Product Details": self.product_details
                }
        dict_ = dict_ | self.summary | self.rewards | self.fees_charges | self.pros_cons

        return dict_


    def to_text(self):
        dict_ =  self.to_dict()
        return json_to_text(dict_)