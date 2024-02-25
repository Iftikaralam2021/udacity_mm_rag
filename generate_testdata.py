from openai import OpenAI
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

questions = [   
                "How big do you want your house to be?" 
                "What are 3 most important things for you in choosing this property?", 
                "Which amenities would you like?", 
                "Which transportation options are important to you?",
                "How urban do you want your neighborhood to be?",   
            ]
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."]

user_prompt = f"""
Generate answers  to the following questions for 5 customers looking for their dream home:
1. How big do you want your house to be?
2. What are 3 most important things for you in choosing this property?
3. Which amenities would you like?
4. Which transportation options are important to you?
5. How urban do you want your neighborhood to be?

Use the example answers below to generate the responses:
1. A comfortable three-bedroom house with a spacious kitchen and a cozy living room.
2. A quiet neighborhood, good local schools, and convenient shopping options.
3. A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.
4. Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.
5. A balance between suburban tranquility and access to urban amenities like restaurants and theaters.

Return the responses in a JSON format, with each key corresponding to a customer and the value being the response to the questions.
Within the main keys, the subkeys should be the questions and the values should be the responses.
For example:
{{"customer1": {{"question1": "response1", "question2": "response2", ...}},
  "customer2": {{"question1": "response1", "question2": "response2", ...}}, ...}}
"""

def get_text_output_test(user_prompt):
    response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are a helpful assistant with insights into customers of a real estate platform looking to purchase a house"},
        {"role": "user", "content": user_prompt}
    ]
    )

    listings = json.loads(response.choices[0].message.content)
    print(listings)
    return listings

if __name__ == '__main__':
    listings = get_text_output_test(user_prompt)
    
    with open('test_data.json', 'w', encoding='utf-8') as f:
        json.dump(listings, f, ensure_ascii=False, indent=4)
    # print(listings)
    # save_text_output(listings,iteration=1)
    # get_images(listings,iteration=1,num_l