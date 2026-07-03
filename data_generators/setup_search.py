from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

from utils import must_get, load_env
from rag import get_clients, embed_text

def main():
    load_env()

    search_endpoint = must_get("SEARCH_ENDPOINT")
    admin_key = must_get("SEARCH_ADMIN_KEY")
    index_name = must_get("SEARCH_INDEX_NAME")

    aoai_endpoint = must_get("AOAI_ENDPOINT")
    aoai_key = must_get("AOAI_KEY")
    api_version = must_get("AOAI_API_VERSION")
    embed_depl = must_get("AOAI_EMBED_DEPLOYMENT")

    # Determine embedding dimension dynamically (important!)
    aoai = get_clients(aoai_endpoint, aoai_key, api_version)
    test_vec = embed_text(aoai, embed_depl, "dimension test")
    dim = len(test_vec)

    idx_client = SearchIndexClient(search_endpoint, AzureKeyCredential(admin_key))

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SearchableField(name="invoice_number", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="vendor_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="customer_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="payment_terms", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="invoice_date", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="due_date", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="currency_code", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="subtotal", type=SearchFieldDataType.Double, filterable=True, facetable=True, sortable=True),
        SimpleField(name="tax_total", type=SearchFieldDataType.Double, filterable=True, facetable=True, sortable=True),
        SimpleField(name="invoice_total", type=SearchFieldDataType.Double, filterable=True, facetable=True, sortable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dim,
            vector_search_profile_name="vprofile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vprofile", algorithm_configuration_name="hnsw")],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

    # Create or update
    idx_client.create_or_update_index(index)
    print(f"✅ Search index ready: {index_name} (embedding dim={dim})")

if __name__ == "__main__":
    main()