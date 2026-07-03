from dataclasses import dataclass
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

@dataclass
class ExtractedChunk:
    page: int
    text: str


INVOICE_FIELD_ALIASES = {
    "invoice_number": ["InvoiceId", "InvoiceNumber", "Invoice #"],
    "vendor_name": ["VendorName", "Vendor", "SellerName"],
    "customer_name": ["CustomerName", "BillTo", "BuyerName"],
    "invoice_date": ["InvoiceDate", "Date"],
    "due_date": ["DueDate"],
    "payment_terms": ["PaymentTerm", "PaymentTerms"],
    "currency_code": ["CurrencyCode"],
    "subtotal": ["SubTotal", "Subtotal"],
    "tax_total": ["TotalTax", "Tax"],
    "invoice_total": ["InvoiceTotal", "Total"],
}

def extract_pdf_text_by_page(
    file_bytes: bytes,
    endpoint: str,
    key: str,
    model_id: str = "prebuilt-layout",
) -> list[ExtractedChunk]:
    """
    Extract text from PDF with page association.
    model_id: "prebuilt-layout" (good for tables/layout) or "prebuilt-read" (OCR heavy).
    """
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = client.begin_analyze_document(
        model_id=model_id,
        body=file_bytes,
        content_type="application/pdf",
    )
    result = poller.result()

    chunks: list[ExtractedChunk] = []
    for page_idx, page in enumerate(result.pages, start=1):
        lines = [line.content for line in (page.lines or []) if line.content]
        page_text = "\n".join(lines).strip()
        if page_text:
            chunks.append(ExtractedChunk(page=page_idx, text=page_text))

    return chunks


def _to_primitive(value):
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [_to_primitive(v) for v in value]

    if isinstance(value, dict):
        return {k: _to_primitive(v) for k, v in value.items()}

    amount = getattr(value, "amount", None)
    if amount is not None:
        return {
            "amount": amount,
            "currency_symbol": getattr(value, "symbol", None),
            "currency_code": getattr(value, "code", None),
        }

    return str(value)


def _field_value(field):
    if field is None:
        return None

    value_string = getattr(field, "value_string", None)
    if value_string is not None:
        return value_string

    value_date = getattr(field, "value_date", None)
    if value_date is not None:
        return str(value_date)

    value_time = getattr(field, "value_time", None)
    if value_time is not None:
        return str(value_time)

    value_float = getattr(field, "value_float", None)
    if value_float is not None:
        return float(value_float)

    value_integer = getattr(field, "value_integer", None)
    if value_integer is not None:
        return int(value_integer)

    value_currency = getattr(field, "value_currency", None)
    if value_currency is not None:
        return _to_primitive(value_currency)

    value_array = getattr(field, "value_array", None)
    if value_array is not None:
        return [_field_value(item) for item in value_array]

    value_object = getattr(field, "value_object", None)
    if value_object is not None:
        return {k: _field_value(v) for k, v in value_object.items()}

    content = getattr(field, "content", None)
    if content is not None:
        return content

    return None


def _pick_field(field_map: dict, aliases: list[str]):
    for name in aliases:
        if name in field_map:
            return field_map[name]
    return None


def _as_amount(value):
    if value is None:
        return None

    if isinstance(value, (float, int)):
        return float(value)

    if isinstance(value, dict):
        amount = value.get("amount")
        if isinstance(amount, (float, int)):
            return float(amount)

    return None


def _normalize_invoice_document(fields: dict) -> dict:
    normalized = {
        "invoice_number": None,
        "vendor_name": None,
        "customer_name": None,
        "invoice_date": None,
        "due_date": None,
        "payment_terms": None,
        "currency_code": None,
        "subtotal": None,
        "tax_total": None,
        "invoice_total": None,
        "line_items": [],
        "field_confidence": {},
        "raw_fields": {},
    }

    for canonical_name, aliases in INVOICE_FIELD_ALIASES.items():
        field = _pick_field(fields, aliases)
        value = _field_value(field)
        conf = getattr(field, "confidence", None) if field else None

        if canonical_name in {"subtotal", "tax_total", "invoice_total"}:
            normalized[canonical_name] = _as_amount(value)
        elif canonical_name == "currency_code":
            if isinstance(value, str):
                normalized[canonical_name] = value
            elif isinstance(value, dict):
                normalized[canonical_name] = value.get("currency_code")
            else:
                normalized[canonical_name] = None
        else:
            normalized[canonical_name] = value

        normalized["field_confidence"][canonical_name] = conf

    items_field = _pick_field(fields, ["Items", "LineItems"])
    items = _field_value(items_field) or []

    for item in items:
        if not isinstance(item, dict):
            continue
        normalized["line_items"].append(
            {
                "description": item.get("Description"),
                "quantity": item.get("Quantity"),
                "unit_price": _as_amount(item.get("UnitPrice")),
                "amount": _as_amount(item.get("Amount")),
                "product_code": item.get("ProductCode"),
            }
        )

    normalized["raw_fields"] = {
        name: {
            "value": _field_value(field),
            "confidence": getattr(field, "confidence", None),
        }
        for name, field in fields.items()
    }

    return normalized


def extract_invoice_data(file_bytes: bytes, endpoint: str, key: str) -> dict:
    """
    Extract normalized invoice data from a PDF using Document Intelligence prebuilt-invoice.
    """
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document(
        model_id="prebuilt-invoice",
        body=file_bytes,
        content_type="application/pdf",
    )
    result = poller.result()

    if not result.documents:
        return {}

    doc = result.documents[0]
    fields = doc.fields or {}
    normalized = _normalize_invoice_document(fields)
    normalized["doc_type"] = getattr(doc, "doc_type", None)

    return normalized