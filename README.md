# DocIntel-Extract
Prototyping Azure AI Document Intelligence for work project, and also AI-102 hands on learning

# Environment Variables
Check `docs/Variables.md` for instructions on how to setup `.env` with the proper variable names

# Important Commands

Generate Dummy pdf data
```bash
python3 generate_dummy_invoice.py
```

Activate Python Virtual Environment
```bash
source venv/bin/activate
```

Run the application
```bash
streamlit run app.py
```

# New AI-102 Hands-On Feature: Retrieval Lab

The app now includes an **AI-102 Retrieval Lab** section in the Q&A panel so you can practice Azure AI Search patterns used in the certification exam:

- `Vector` retrieval mode (embedding-only)
- `Hybrid` retrieval mode (keyword + vector)
- Optional semantic ranking for hybrid search
- Debug view with `@search.score` and `@search.reranker_score`

Suggested practice flow:

1. Index one PDF and ask the same question in `Vector` mode.
2. Switch to `Hybrid` mode and compare answer quality/citations.
3. Enable semantic ranking and compare reranker scores.
4. Use the retrieved chunk debug panel to understand ranking behavior.

Detailed guide with concepts and examples:

- `docs/AI102_Retrieval_Lab_README.md`
- `docs/AI102_Evaluation_Lab_README.md`

![UI](./photos/image.png)

![Answer](./photos/image-1.png)