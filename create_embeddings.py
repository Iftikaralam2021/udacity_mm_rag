
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import lancedb
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
uri = "./sample-lancedb"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = "openai/clip-vit-base-patch32"

def create_clip_image_embeddings(image_path, model_name):
    # Load the image
    image = Image.open(image_path)

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Generate the image embedding
    inputs = processor(images=image, return_tensors="pt")

    image_embedding = model.get_image_features(**inputs)

    # Return the image embedding
    return image_embedding.detach().numpy()[0]

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding





if __name__ == '__main__':

    df = pd.read_csv('listings.csv')
    # Create the full description  column  combining all the text and data  columns which will be used to generate the embeddings
    df["full_description"] = "Neighbourhood: " +df["Neighborhood"] + \
        " Price: " + df["Price"].astype(str) + " Bedrooms: " + df["Bedrooms"].astype(str) + \
        " Bathrooms: " + df["Bathrooms"].astype(str) + " House Size: " + df["House Size"].astype(str) + \
       "Description: " + df["Description"] + " " + "Neighborhood Description: " + df["Neighborhood Description"]
    df['ada_embedding'] = df["full_description"].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    df['image_embedding'] = df["images"].apply(lambda x: create_clip_image_embeddings(x, model_name))
    df.to_csv('listings_with_embeddings.csv', index=False)
    db = lancedb.connect(uri)

    df_text = df[["Neighborhood","Price","Bedrooms","Bathrooms","images","ada_embedding"]].copy()
    df_text = df_text.rename(columns={"ada_embedding":"vector"})
    df_images = df[["Neighborhood","Price","Bedrooms","Bathrooms","images","image_embedding"]].copy()
    df_images = df_images.rename(columns={"image_embedding":"vector"})

    tbl_text = db.create_table("table_from_df_text", data=df_text, exist_ok=True)
    print(df_images.head())
    print(df_images["vector"][0].shape)
    tbl_images = db.create_table("table_from_df_images", data=df_images,exist_ok=True)


