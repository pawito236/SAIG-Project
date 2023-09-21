import os
import pinecone
import numpy as np
import openai
import json

with open('keys.json', 'r') as json_file:
    data_keys = json.load(json_file)

os.environ["OPENAI_API_KEY"] = data_keys["OPENAI_API_KEY"]
openai.api_key = data_keys["OPENAI_API_KEY"]

pinecone.init(api_key=data_keys["pinecone"], environment="gcp-starter")
index = pinecone.Index("saig-project")


docs = []
DIR = "docs/"
for path in os.listdir(DIR):
  with open(DIR+path) as f:
      contents = f.read()
      docs.append(contents)

pinecone_vectors = []
for i in range(len(docs)):
    entry = docs[i]
    embedding = openai.Embedding.create(
        input=entry,
        model="text-embedding-ada-002"
    )

    # print the embedding (length = 1536)
    vector = embedding["data"][0]["embedding"]

    # append tuple to pinecone_vectors list
    pinecone_vectors.append((str(i), vector, {"context": docs[i]}))

# delete_response = index.delete(ids=[str(i) for i in range(len(docs))])
upsert_response = index.upsert(vectors=pinecone_vectors)
print("Vector upload complete.")