# Architecture & Model Selection

## System Overview

```mermaid
graph TD
    subgraph Input
        A[PDF Upload]
        B[Natural Language Query]
        C[Manual Parameters]
    end

    subgraph Processing
        A -->|PyMuPDF| D[Text Extraction]
        D -->|Claude API| E[Structured Data - JSON]

        B -->|sentence-transformers| F[Query Embedding]
        F -->|ChromaDB| G[Similarity Search]
        G -->|Claude API| H[Intelligent Ranking]

        E -->|Parameter Mapping| I[NFC Config Generator]
        C --> I

        E -->|Features| J[RandomForest Regressor]
    end

    subgraph Product Knowledge
        K[products.json] -->|Embedding| L[ChromaDB Vector Store]
        L --> G
        K --> I
    end

    subgraph Output
        E --> M[PDF Report]
        H --> M
        I --> M
        J --> M
        M -->|fpdf2| N[Download]
    end

    subgraph Evaluation
        D -->|Same Input| O[Claude Sonnet]
        D -->|Same Input| P[Llama-3.3-70B]
        O --> Q[Comparison Table]
        P --> Q
    end
```

## Model Selection Rationale

### LLM: Claude Sonnet 4.6 (Primary)
- **Strength:** Most reliable JSON output format, excellent multilingual competence
- **Trade-off:** Higher cost than open-source, API dependency
- **Alternative evaluated:** Llama-3.3-70B via Groq — faster/cheaper, but less reliable for structured extraction

### Embeddings: sentence-transformers/all-MiniLM-L6-v2
- **Strength:** Local, free, no API latency, no vendor lock-in
- **Trade-off:** Not specialized for domain-specific HVAC terminology
- **Alternatives:** voyage-3, text-embedding-3-large — better quality, but API-dependent

### Vector Store: ChromaDB
- **Strength:** In-memory, no server, persistent-capable, Python-native
- **Trade-off:** Not scalable to millions of documents
- **Sufficient:** 15 products → In-memory ChromaDB is optimal

### ML: RandomForest Regressor
- **Strength:** Interpretable (feature importance), robust with small datasets (n=60), no overfitting risk
- **Trade-off:** No extrapolation beyond training range
- **Why not Neural Networks:** With n=60 synthetic samples, any NN would be massively overfitted

### PDF Parsing: PyMuPDF
- **Strength:** Fastest Python library, reliable, open source
- **Alternative:** pdfplumber (better table support), but slower

## Data Flow

1. **PDF Upload** → PyMuPDF extracts raw text
2. **Claude API** → Extracts structured JSON from raw text
3. **ChromaDB** → Products are embedded and indexed at startup
4. **Semantic Search** → User query is embedded, compared against product vectors
5. **Claude Ranking** → Top candidates are re-ranked by LLM with reasoning
6. **Energy Model** → RandomForest predicts annual savings from room parameters
7. **PDF Report** → All results compiled into downloadable PDF via fpdf2
