import streamlit as st
import pandas as pd
import requests
import json 

if __name__ == "__main__":
    import sys
    sys.path.append('code')
    from apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition
else:
    from code.apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition

PLACE_IDS_SOURCE_FILE = "cache/place_ids.csv"
CACHE_REVIEWS_FILE = "cache/reviews.csv"
CACHE_SENTIMENT_FILE = "cache/reviews_sentiment_by_sentence.csv"
CACHE_ENTITIES_FILE = "cache/reviews_sentiment_by_sentence_with_entities.csv"

def reviews_step(place_ids: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(place_ids, str):
        df_ids = pd.read_csv(place_ids)
    else:
        df_ids = place_ids

    results = []
    for _, row in df_ids.iterrows():
        details = get_google_place_details(row["Google Place ID"])
        results.append(details["result"])

    df_reviews = pd.json_normalize(results, record_path="reviews", meta=["place_id", "name"])
    df_reviews = df_reviews[["place_id", "name", "author_name", "rating", "text"]]

    df_reviews = df_reviews.head(10)  # force exactly 10 rows
    df_reviews.to_csv(CACHE_REVIEWS_FILE, index=False)
    return df_reviews

def sentiment_step(reviews: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(reviews, str):
        df = pd.read_csv(reviews)
    else:
        df = reviews

    output = []
    for _, row in df.iterrows():
        response = get_azure_sentiment(row["text"])
        doc = response["results"]["documents"][0]
        doc.update({
            "place_id": row["place_id"],
            "name": row["name"],
            "author_name": row["author_name"],
            "rating": row["rating"]
        })
        output.append(doc)

    df_sentences = pd.json_normalize(output, record_path="sentences", meta=["place_id", "name", "author_name", "rating"])
    df_sentences.rename(columns={
        "text": "sentence_text",
        "sentiment": "sentence_sentiment"
    }, inplace=True)

    df_sentences = df_sentences[[
        "place_id", "name", "author_name", "rating", 
        "sentence_text", "sentence_sentiment", 
        "confidenceScores.positive", "confidenceScores.neutral", "confidenceScores.negative"
    ]]

    df_sentences.to_csv(CACHE_SENTIMENT_FILE, index=False)
    return df_sentences

def entity_extraction_step(sentiment: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(sentiment, str):
        df = pd.read_csv(sentiment)
    else:
        df = sentiment

    all_entities = []
    for _, row in df.iterrows():
        response = get_azure_named_entity_recognition(row["sentence_text"])
        doc = response["results"]["documents"][0]
        for col in df.columns:
            doc[col] = row[col]
        all_entities.append(doc)

    df_entities = pd.json_normalize(all_entities, record_path="entities", meta=list(df.columns))

    df_entities.rename(columns={
        "text": "entity_text",
        "category": "entity_category",
        "subcategory": "entity_subcategory",
        "confidenceScore": "confidenceScores.entity"
    }, inplace=True)

    df_entities = df_entities[[
        "place_id", "name", "author_name", "rating", 
        "sentence_text", "sentence_sentiment", 
        "confidenceScores.positive", "confidenceScores.neutral", "confidenceScores.negative",
        "entity_text", "entity_category", "entity_subcategory", "confidenceScores.entity"
    ]]

    df_entities.to_csv(CACHE_ENTITIES_FILE, index=False)
    return df_entities

if __name__ == '__main__':
    st.write("What do you want to debug?")
