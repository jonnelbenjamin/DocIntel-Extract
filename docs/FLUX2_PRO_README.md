# FLUX.2-Pro in This Project (Azure AI Foundry)

This document explains what FLUX.2-Pro does, where it fits in this codebase, and how to use it from the Streamlit app.

## What FLUX.2-Pro Is

FLUX.2-Pro is a high-quality text-to-image model available through Azure AI Foundry model deployments.

In practical terms, it is used for:
- text prompt → generated image
- photorealistic scenes
- stylized art direction
- concept and visual ideation

## What It Does Well

- Strong prompt adherence for scene composition
- Good detail quality at standard image sizes
- Useful for rapid creative iteration in product or content workflows

## Typical Limitations

- Prompt phrasing still matters a lot
- Complex text rendering inside images can be imperfect
- Output can vary between runs even with similar prompts
- Safety filters may block sensitive prompts

## How It Is Integrated Here

Files:
- `flux.py`
  - `generate_flux_image(...)`: calls the Azure endpoint
  - `extract_flux_image_bytes(...)`: supports base64 and URL responses

- `app.py`
  - New section: `4) FLUX.2-Pro — Image Generation (Azure AI Foundry)`
  - UI fields for endpoint, key, deployment, API version
  - Prompt + size + output format
  - Generated image preview + file download

## Required Configuration

Set these in your `.env` file (or enter directly in the app section):

```env
FLUX_ENDPOINT=https://<your-resource>.openai.azure.com
FLUX_KEY=<your-api-key>
FLUX_DEPLOYMENT=flux-2-pro
FLUX_API_VERSION=2024-05-01-preview
```

Notes:
- Endpoint/key must match the resource where FLUX.2-Pro is deployed.
- If your tenant requires a different API version, use the one shown in Azure Foundry for your deployment.

### Endpoint formats supported by this app

The app supports **both** formats:

1. Azure OpenAI style:

```env
FLUX_ENDPOINT=https://<resource>.openai.azure.com
FLUX_DEPLOYMENT=flux-2-pro
FLUX_API_VERSION=2024-05-01-preview
```

2. Azure AI Foundry provider style (full model endpoint):

```env
FLUX_ENDPOINT=https://<project>.services.ai.azure.com/providers/blackforestlabs/v1/flux-2-pro?api-version=preview
FLUX_API_VERSION=preview
```

When using provider style endpoint, deployment name is embedded in the URL and `FLUX_DEPLOYMENT` is ignored.

## Prompting Tips

To get better results:
- include subject + setting + style + lighting + camera/framing
- specify quality descriptors (for example: cinematic, high detail, realistic)
- avoid overly ambiguous prompts when precision is required

Example:

`Cinematic product photo of a matte-black smartwatch on marble, soft studio lighting, shallow depth of field, ultra-detailed.`

## Run and Validate

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

3. Open section `4) FLUX.2-Pro — Image Generation (Azure AI Foundry)`.
4. Enter endpoint/key/deployment/api version and a prompt.
5. Click `Generate FLUX image`.

If successful, image preview and download appear in the UI.
