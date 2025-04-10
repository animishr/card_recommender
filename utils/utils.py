KEYS = ['Product Name', 
        'Product Description', 
        'Product Details', 
        'Joining Fee', 
        'Renewal Fee', 
        'Best Suited For', 
        'Reward Type', 
        'Welcome Benefits', 
        'Movie & Dining', 
        'Travel', 
        'Domestic Lounge Access', 
        'International Lounge Access', 
        'Golf', 
        'Insurance Benefits', 
        'Spend-Based Waiver', 
        'Foreign Currency Markup', 
        'Interest Rates', 
        'Fuel Surcharge']

ITERABLE_KEYS = ['Product Details', 
                 'Best Suited For', 
                 'Reward Type']


def json_to_text(json_):
    text = ""
    for key in KEYS:
        value = json_[key] if json_[key] else ""
        if key in ITERABLE_KEYS:
            text += f"** {key}: **"
            for element in value:
                text += f"\n\t* {element}"
            text += "\n"
        else:
            text += f"** {key}: ** {value}\n"
    
    return text