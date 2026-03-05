from openai import AzureOpenAI

def get_clients(aoai_endpoint: str, aoai_key: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        api_version=api_version,
    )

def embed_text(client: AzureOpenAI, embed_deployment: str, text: str) -> list[float]:
    resp = client.embeddings.create(
        model=embed_deployment,
        input=text,
    )
    return resp.data[0].embedding

def chat_answer_with_citations(
    client: AzureOpenAI,
    chat_deployment: str,
    question: str,
    retrieved: list[dict],
) -> str:
    """
    retrieved items should include: { "content": str, "source": str, "page": int }
    We'll build a grounded prompt and ask for citations.
    """
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[{i}] Source: {item.get('source')} | Page: {item.get('page')}\n{item.get('content')}"
        )
    context = "\n\n".join(context_blocks)

    system = (
        "You are a careful assistant. Use ONLY the provided context. "
        "If the answer isn't in the context, say you don't know. "
        "When you use a fact, cite it like [1], [2], etc."
    )

    user = f"""Question: {question}

Context:
{context}

Answer with citations:"""

    resp = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content