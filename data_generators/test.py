from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import os

load_dotenv()

print("endpoint:", os.getenv("AOAI_ENDPOINT"))

client = AzureOpenAI(
    azure_endpoint=os.environ["AOAI_ENDPOINT"],
    api_key=os.environ["AOAI_KEY"],
    api_version=os.environ["AOAI_API_VERSION"],
)

# chat
chat = client.chat.completions.create(
    model=os.environ["AOAI_CHAT_DEPLOYMENT"],
    messages=[{"role":"user","content":"Say 'chat ok'"}],
)
print(chat.choices[0].message.content)

# embeddings
emb = client.embeddings.create(
    model=os.environ["AOAI_EMBED_DEPLOYMENT"],
    input="embedding ok",
)
print(len(emb.data[0].embedding))