from openai import OpenAI
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

client = OpenAI(api_key='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')


example_listing = """Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
"""



def get_text_output(user_prompt):
    response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are a helpful assistant with deep expertise in real estate."},
        {"role": "user", "content": user_prompt}
    ]
    )

    listings = json.loads(response.choices[0].message.content)["listings"]
    # print(listings)
    return listings

def save_text_output(listings,iteration=1):
    os.makedirs("listings", exist_ok=True)
    text_dataframe = pd.DataFrame(listings)
    text_dataframe.to_csv(f"listings/listings_{iteration}.csv", index=False)
        

def get_images(listings,iteration=1,num_listings=5):
    os.makedirs("images", exist_ok=True)
    for k,listing in enumerate(listings):
        prompt = f"Generate a photorealistic image of a house whose description is: {listing['Description']}. The image should look like a photograph with muted colors and will be posted on a real estate website."
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        img = requests.get(image_url).content
        # i = Image.open(BytesIO(img))
        # plt.figure()
        # plt.imshow(i)
        # plt.show()
        k=k+(iteration-1)*num_listings
        with open(f"images/image_{k}.png", "wb") as f:
            f.write(img)


if __name__ == "__main__":
    num_listings = 5
    