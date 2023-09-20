import os
import pinecone
import numpy as np
import openai

pinecone.init(api_key="", environment="gcp-starter")
index = pinecone.Index("qa")
index