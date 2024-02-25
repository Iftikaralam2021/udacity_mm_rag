import json
from generate_data import *
from create_embeddings import *
import lancedb


uri = "./sample-lancedb"
db = lancedb.connect(uri)

text_table = "table_from_df_text"
img_table = "table_from_df_images"

tbl_txt = db.open_table(text_table)
tbl_img = db.open_table(img_table)


with open('./test_data.json') as f:
    test_user_scripts = json.loads(f.read())

customers = list(test_user_scripts.keys())

customer_id = 0 # change this to test for different customers
# Format user chat history
def build_customer_chat_history(customer_id):
    outstring=""
    for q,a in test_user_scripts[customers[customer_id]].items():
        outstring+=f"Question- {q} : User Answer: {a}"
    return outstring

# this function reformats the user inputs to a json structured output very similar to the one used to create embeddings

def get_reformatted_output(user_prompt):
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

def format_response(response):
    out_string=""
    for item,val in response.items():
        if val != 'None':
            out_string+=f"{item}: {val}  "
    return out_string

def get_user_preference(customer_id, img_path=None):
    # if user provides an image as reference, we shall also use that. The assumption is that the image has been 
    # loaded and placed in a path the application can access
    image = img_path
    # if img_path:
    #     try: 
    #         image = Image.open(image_path)
    #     except:
    #         pass

    chat_history =  build_customer_chat_history(customer_id) 
    user_prompt= f"""
    Please only use the customer chat history given below to create a desired listing for them. 
    Use the example given below and  format the results in json format.
    All the results should be saved inside a key called listings.
    Each result should have the following keys: Neighborhood, Price, Bedrooms, Bathrooms, House Size, Description, Neighborhood Description.
    Use only information from the chat history. If any of the fields are unavailable,list them as None.
    Customer Chat History: {chat_history}
    Example:{example_listing}
    """
    response = get_reformatted_output(user_prompt)[0]
    formatted_response = format_response(response)
    return formatted_response, image


def get_embeddings_user_prefs(resp):
    text_resp, img_resp = resp[0],resp[1]
    text_embs = get_embedding(text_resp)
    img_embs = None
    if img_resp:
        try:
            img_embs = create_clip_image_embeddings(img_resp, model_name)
        except:
            pass

    return text_embs,img_embs

def search_tables(embeddings,num_responses=5):
    text_embeddings = embeddings[0]
    img_embeddings = embeddings[1]
    df = tbl_txt.search(text_embeddings) \
    .metric("cosine") \
    .limit(num_responses) \
    .to_pandas()
    return df

resp = get_user_preference(customer_id)
embeddings = get_embeddings_user_prefs(resp)

print(search_tables(embeddings))




