Create a .env file in the root of the project and add the following variables.

Replace the placeholder values with your own Azure service credentials.

## ---------- Azure Document Intelligence ----------
```bash
DOCINTEL_ENDPOINT=your_document_intelligence_endpoint
DOCINTEL_KEY=your_document_intelligence_key
```

## ---------- Azure AI Search ----------
```bash
SEARCH_ENDPOINT=your_ai_search_endpoint
SEARCH_ADMIN_KEY=your_ai_search_admin_key
SEARCH_INDEX_NAME=your_ai_search_index_name
# Optional for semantic ranking exercises in Retrieval Lab UI
# SEARCH_SEMANTIC_CONFIG=default
```

## ---------- Azure OpenAI ----------
```bash
AOAI_ENDPOINT=your_azure_openai_endpoint
AOAI_KEY=your_azure_openai_key
AOAI_API_VERSION=2024-12-01-preview
```

## Deployments created in Azure OpenAI / Azure AI Foundry
```bash
AOAI_CHAT_DEPLOYMENT=your_chat_model_deployment_name
AOAI_EMBED_DEPLOYMENT=your_embedding_model_deployment_name
```

Where to Find These Values
### Variable	        Where to get it
---
- DOCINTEL_ENDPOINT	
    > Azure Portal → Document Intelligence → Keys and Endpoint
- DOCINTEL_KEY	
    > Azure Portal → Document Intelligence → Keys and Endpoint
- SEARCH_ENDPOINT	
    > Azure Portal → AI Search → Overview
- SEARCH_ADMIN_KEY	
    > Azure Portal → AI Search → Keys
- SEARCH_INDEX_NAME	
    > The index you created in Azure AI Search
- AOAI_ENDPOINT	
    > Azure Portal → Azure OpenAI → Keys and Endpoint
- AOAI_KEY	
    > Azure Portal → Azure OpenAI → Keys and Endpoint
- AOAI_API_VERSION	
    > Azure OpenAI API version used by the SDK
- AOAI_CHAT_DEPLOYMENT	
    > Name of your deployed chat model (ex: gpt-4.1-mini)
- AOAI_EMBED_DEPLOYMENT	
    > Name of your embedding deployment (ex: text-embedding-3-large)