"""
DocAI Assistant — AI-powered document analysis for HVAC equipment.
Upload PDFs, extract structured data, search products semantically, generate reports.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from sklearn.ensemble import RandomForestRegressor

load_dotenv()

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_PATH = BASE_DIR / "products.json"
DEMO_EXTRACTION = BASE_DIR / "demo_data" / "demo_extraction_result.json"
DEMO_SEARCH = BASE_DIR / "demo_data" / "demo_search_results.json"
DEMO_COMPARISON = BASE_DIR / "demo_data" / "demo_comparison.json"
SAMPLE_PDF = BASE_DIR / "demo_data" / "sample_datasheet.pdf"
PROMPT_EXTRACTION = BASE_DIR / "prompts" / "extraction_system.md"
PROMPT_MATCHING = BASE_DIR / "prompts" / "matching_system.md"

ACCENT_COLOR = "#2563EB"
MAX_EXTRACTION_CHARS = 15000
MAX_PDF_SIZE_MB = 20
MAX_PDF_PAGES = 100
DE_CO2_EMISSION_FACTOR_KG_PER_KWH = 0.4
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_API_CALLS_PER_SESSION = 20

ELS_NFC_PARAMS = {
    "models": ["ELS NFC", "ELS NFC F", "ELS NFC P", "ELS NFC VOC", "ELS NFC CO2"],
    "airflow_steps_m3h": [7.5, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
    "factory_defaults_m3h": {"stage_1": 35, "stage_2": 60, "stage_3": 100},
    "max_stages": 5,
    "delay_range_sec": {"min": 0, "max": 120},
    "runon_range_min": {"min": 0, "max": 90},
    "interval_range_h": {"min": 0, "max": 24},
    "basic_ventilation_m3h": 15,
    "sensor_options": {
        "ELS NFC F": {"type": "humidity", "unit": "%rH", "threshold_adjustable": True},
        "ELS NFC P": {"type": "presence", "detection": "PIR"},
        "ELS NFC VOC": {
            "type": "voc",
            "unit": "voc",
            "threshold_range": [100, 450],
            "max_value_range": [100, 450],
            "modes": ["Comfort", "Intensive"],
        },
        "ELS NFC CO2": {"type": "co2", "unit": "ppm", "threshold_adjustable": True},
    },
    "nfc_features": {
        "offline_config": True,
        "library_support": True,
        "status_readout": True,
        "error_reporting": True,
    },
}

ARTICLE_NUMBERS = {
    "ELS NFC": "40761",
    "ELS NFC F": "40762",
    "ELS NFC P": "40763",
    "ELS NFC VOC": "40764",
    "ELS NFC CO2": "40765",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str | None:
    """Return Anthropic API key from sidebar input, env, or None."""
    key = st.session_state.get("anthropic_api_key", "")
    if key:
        return key
    return os.getenv("ANTHROPIC_API_KEY")


def load_json(path: Path) -> dict | None:
    """Load JSON file with error handling. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {path.name}")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON file: {path.name}")
        return None
    except Exception:
        st.error(f"Error loading {path.name}")
        return None


def read_prompt(path: Path) -> str:
    """Read prompt file with error handling. Returns empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"Prompt file not found: {path.name}")
        return ""
    except Exception:
        st.warning(f"Error loading {path.name}")
        return ""


def safe_latin1(text: str) -> str:
    """Encode text to latin-1 safe form for PDF Helvetica font."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


@st.cache_resource
def load_product_catalog() -> dict | None:
    try:
        return load_json(PRODUCTS_PATH)
    except Exception:
        st.error("Product catalog could not be loaded.")
        return None


@st.cache_resource
def build_chroma_collection():
    """Embed products and store in ChromaDB in-memory collection."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        catalog = load_json(PRODUCTS_PATH)
        if catalog is None:
            return None, None

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.Client()

        try:
            client.delete_collection("products")
        except Exception:
            pass

        collection = client.create_collection(
            name="products", metadata={"hnsw:space": "cosine"}
        )

        ids = []
        documents = []
        metadatas = []

        for p in catalog["products"]:
            doc_text = _product_to_text(p)
            ids.append(p["id"])
            documents.append(doc_text)
            metadatas.append(
                {
                    "name": p["name"],
                    "category": p["category"],
                    "nfc": str(p["specs"].get("nfc_configurable", False)),
                    "use_cases": ", ".join(p.get("use_cases", [])),
                }
            )

        embeddings = model.encode(documents).tolist()
        collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        return collection, model
    except Exception:
        st.error("ChromaDB setup failed. Please reload the page.")
        return None, None


def _product_to_text(p: dict) -> str:
    """Create a searchable text representation of a product."""
    specs = p.get("specs", {})
    parts = [
        p["name"],
        p.get("category", ""),
        p.get("description", ""),
        f"Use cases: {', '.join(p.get('use_cases', []))}",
        f"Highlights: {', '.join(p.get('highlights', []))}",
    ]
    airflow = specs.get("airflow_m3h")
    if airflow is not None:
        if isinstance(airflow, list):
            parts.append(f"Airflow: {min(airflow)}-{max(airflow)} m\u00b3/h")
        else:
            parts.append(f"Airflow: {airflow} m\u00b3/h")

    sound = specs.get("sound_pressure_dba")
    if isinstance(sound, dict):
        parts.append(
            f"Sound level: {sound.get('low', '?')}-{sound.get('high', '?')} dB(A)"
        )

    if specs.get("sensor"):
        parts.append(f"Sensor: {specs['sensor']}")
    if specs.get("nfc_configurable"):
        parts.append("NFC configuration available")
    if specs.get("wrg_efficiency_pct"):
        parts.append(f"HRV efficiency: {specs['wrg_efficiency_pct']}%")
    if specs.get("ex_rating"):
        parts.append(f"Explosion protection: {specs['ex_rating']}")

    return " | ".join(parts)


CLAUDE_MODEL_FALLBACK = "claude-sonnet-4-20250514"


def call_claude(system_prompt: str, user_content: str) -> str | None:
    """Call Anthropic Claude API with error handling and model fallback."""
    api_key = get_api_key()
    if not api_key:
        return None

    call_count = st.session_state.get("api_call_count", 0)
    if call_count >= MAX_API_CALLS_PER_SESSION:
        st.warning(
            f"API limit reached ({MAX_API_CALLS_PER_SESSION} calls per session). "
            "Reload the page for a new session."
        )
        return None

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    models_to_try = [CLAUDE_MODEL, CLAUDE_MODEL_FALLBACK]
    last_error = None

    for model_id in models_to_try:
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )

            st.session_state["api_call_count"] = call_count + 1
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            total_cost = st.session_state.get("api_total_cost_usd", 0.0)
            total_cost += (input_tokens * 3 + output_tokens * 15) / 1_000_000
            st.session_state["api_total_cost_usd"] = total_cost

            block = response.content[0]
            return getattr(block, "text", None)

        except (anthropic.NotFoundError, anthropic.BadRequestError):
            last_error = f"Model '{model_id}' not available."
            continue
        except anthropic.AuthenticationError:
            st.error(
                "Invalid API key. Please check your Anthropic API key "
                "in the sidebar."
            )
            return None
        except anthropic.RateLimitError:
            st.warning(
                "API rate limit reached. Please wait a moment "
                "and try again."
            )
            return None
        except anthropic.APIConnectionError:
            st.error(
                "Connection to Anthropic API failed. "
                "Please check your internet connection."
            )
            return None
        except anthropic.APIStatusError as e:
            st.error(f"API error (status {e.status_code}). Please try again later.")
            return None
        except Exception:
            st.warning("Unexpected error during API call. Please try again.")
            return None

    if last_error:
        st.error(
            f"No supported model available ({', '.join(models_to_try)}). "
            "Please check your API access."
        )
    return None


def generate_training_data(n: int = 60) -> pd.DataFrame:
    """Generate synthetic training data for energy savings prediction."""
    rng = np.random.default_rng(42)
    room_size = rng.uniform(20, 300, n)
    air_changes = rng.uniform(0.5, 5.0, n)
    ceiling_height = rng.uniform(2.4, 4.0, n)
    wrg_efficiency = rng.uniform(0.5, 0.95, n)
    hours_per_day = rng.uniform(6, 24, n)
    delta_t = rng.uniform(10, 30, n)
    heating_days = rng.uniform(180, 240, n)

    volume = room_size * ceiling_height
    energy_saved_kwh = (
        volume
        * air_changes
        * 0.34
        * delta_t
        * wrg_efficiency
        * hours_per_day
        * heating_days
        / 1000
    )
    energy_saved_kwh *= rng.uniform(0.85, 1.15, n)

    return pd.DataFrame(
        {
            "room_size_m2": room_size,
            "air_changes_per_h": air_changes,
            "ceiling_height_m": ceiling_height,
            "wrg_efficiency_pct": wrg_efficiency * 100,
            "hours_per_day": hours_per_day,
            "delta_t_k": delta_t,
            "heating_days": heating_days,
            "energy_saved_kwh": energy_saved_kwh,
        }
    )


@st.cache_resource
def train_energy_model():
    """Train a RandomForest model for energy savings estimation."""
    df = generate_training_data(60)
    feature_cols = [
        "room_size_m2",
        "air_changes_per_h",
        "ceiling_height_m",
        "wrg_efficiency_pct",
        "hours_per_day",
        "delta_t_k",
        "heating_days",
    ]
    X = df[feature_cols]
    y = df["energy_saved_kwh"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, feature_cols


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------


class DocAIReport(FPDF):
    ACCENT_RGB = (37, 99, 235)
    TEXT_BLACK = (26, 26, 26)
    BG_GRAY = (245, 245, 245)

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.ACCENT_RGB)
        self.cell(0, 10, safe_latin1("DocAI Assistant - Analysis Report"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.ACCENT_RGB)
        self.line(10, 22, 200, 22)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(
            0, 10,
            safe_latin1(f"Generated by DocAI Assistant | Page {self.page_no()}"),
            align="C",
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.ACCENT_RGB)
        self.cell(0, 8, safe_latin1(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.TEXT_BLACK)
        self.set_font("Helvetica", "", 10)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.TEXT_BLACK)
        self.multi_cell(0, 5, safe_latin1(text))
        self.ln(2)

    def key_value_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(60, 6, safe_latin1(key), border=0)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, safe_latin1(str(value)), border=0, new_x="LMARGIN", new_y="NEXT")


def build_pdf_report() -> bytes:
    """Build a PDF report from all available session data."""
    pdf = DocAIReport()
    pdf.add_page()

    pdf.body_text(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    extracted = st.session_state.get("extracted_data")
    if extracted:
        pdf.section_title("1. Extracted Data")
        pdf.body_text(
            f"Document type: {extracted.get('document_type', 'N/A')} | "
            f"Confidence: {extracted.get('confidence', 'N/A')}"
        )
        for product in extracted.get("products", []):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, safe_latin1(product.get("product_name", "Unknown")), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            for k, v in product.items():
                if k != "product_name" and v is not None:
                    pdf.key_value_row(k, str(v))
            pdf.ln(3)

    search = st.session_state.get("search_results")
    if search:
        pdf.section_title("2. Product Recommendations")
        for rec in search.get("recommendations", [])[:3]:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(
                0, 7,
                safe_latin1(f"#{rec['rank']} {rec['product_name']} (Score: {rec['score']})"),
                new_x="LMARGIN", new_y="NEXT",
            )
            pdf.body_text(rec.get("reasoning", ""))

    nfc = st.session_state.get("nfc_config")
    if nfc:
        pdf.section_title("3. NFC Configuration")
        pdf.set_font("Courier", "", 9)
        nfc_text = json.dumps(nfc, indent=2, ensure_ascii=False)
        pdf.multi_cell(0, 4, safe_latin1(nfc_text))
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(3)

    energy = st.session_state.get("energy_result")
    if energy:
        pdf.section_title("4. Energy Savings")
        pdf.body_text(
            f"Estimated annual savings: {energy.get('kwh', 0):,.0f} kWh/year"
        )
        pdf.body_text(
            f"Equivalent to approx. {energy.get('euro', 0):,.0f} EUR/year "
            f"(at {energy.get('price_ct', 30)} ct/kWh)"
        )

    pdf.section_title("5. Methodology")
    pdf.body_text(
        "LLM: Claude Sonnet 4.6 (Anthropic) for structured extraction and ranking. "
        "Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local). "
        "Vector Store: ChromaDB (in-memory). "
        "ML: RandomForest Regressor (scikit-learn) on synthetic data (n=60). "
        "PDF Parsing: PyMuPDF."
    )
    pdf.body_text(
        "Note: This report was generated by a prototype and serves "
        "as a concept demonstration. Not a planning basis."
    )

    output = pdf.output()
    if output is None:
        return b""
    return bytes(output)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _test_api_connection(api_key: str):
    """Test API key and model availability with a minimal request."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    models_to_try = [CLAUDE_MODEL, CLAUDE_MODEL_FALLBACK]

    for model_id in models_to_try:
        try:
            client.messages.create(
                model=model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            st.success(f"Connection OK! Model: {model_id}")
            return
        except anthropic.NotFoundError:
            continue
        except anthropic.BadRequestError:
            continue
        except anthropic.AuthenticationError:
            st.error("API key is invalid. Please check.")
            return
        except anthropic.RateLimitError:
            st.warning("Rate limit reached. Key is valid, but please wait.")
            return
        except anthropic.APIConnectionError:
            st.error("No connection to API. Check internet.")
            return
        except anthropic.APIStatusError as e:
            st.error(f"API error (status {e.status_code}).")
            return
        except Exception:
            st.error("Unexpected error during connection test.")
            return

    st.error(
        f"No supported model available. "
        f"Tested: {', '.join(models_to_try)}",
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### DocAI Assistant")
        st.markdown(
            "AI-powered document analysis for HVAC equipment. "
            "Upload PDFs, extract data, search products semantically."
        )

        st.divider()
        st.markdown("**Features:**")
        st.markdown(
            "- PDF Data Extraction\n"
            "- Semantic Product Search\n"
            "- NFC Configuration\n"
            "- Energy Estimation\n"
            "- Model Evaluation\n"
            "- PDF Report"
        )

        st.divider()
        st.text_input(
            "Anthropic API Key (optional)",
            type="password",
            key="anthropic_api_key",
            help="Not stored. Required for live AI features.",
        )

        api_status = get_api_key()
        if api_status:
            st.success("API key provided")
            call_count = st.session_state.get("api_call_count", 0)
            total_cost = st.session_state.get("api_total_cost_usd", 0.0)
            st.caption(
                f"Calls: {call_count}/{MAX_API_CALLS_PER_SESSION} | "
                f"Cost: ~${total_cost:.3f}"
            )
            if st.button("Test API connection", key="test_api"):
                _test_api_connection(api_status)
        else:
            st.info("Demo mode active")

        st.divider()
        st.caption("Built with Streamlit + Claude API")


# ---------------------------------------------------------------------------
# Tab 1: PDF Extraction
# ---------------------------------------------------------------------------


def _process_pdf_bytes(pdf_bytes: bytes) -> None:
    """Extract text from PDF bytes and run Claude extraction."""
    import fitz

    if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
        st.error(f"PDF too large (max. {MAX_PDF_SIZE_MB} MB).")
        return

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count > MAX_PDF_PAGES:
        st.warning(
            f"PDF has {doc.page_count} pages. "
            f"Only the first {MAX_PDF_PAGES} will be processed."
        )

    pages = [doc[i] for i in range(min(doc.page_count, MAX_PDF_PAGES))]
    raw_text = "".join(str(page.get_text()) for page in pages)
    doc.close()

    if not raw_text.strip():
        st.warning("No text found in PDF. May be a scanned/image PDF.")
        return

    with st.expander("Show raw text", expanded=False):
        st.text_area("Extracted text", raw_text, height=200, disabled=True)

    if len(raw_text) > MAX_EXTRACTION_CHARS:
        st.info(
            f"Document has {len(raw_text):,} characters. "
            f"First {MAX_EXTRACTION_CHARS:,} will be used for extraction."
        )

    api_key = get_api_key()
    if api_key:
        with st.spinner("Analyzing document..."):
            system_prompt = read_prompt(PROMPT_EXTRACTION)
            if not system_prompt:
                st.warning("Extraction prompt not available. Showing demo data.")
                _load_demo_extraction()
                return
            result_text = call_claude(system_prompt, raw_text[:MAX_EXTRACTION_CHARS])

        if result_text:
            try:
                cleaned = result_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                    cleaned = cleaned.rsplit("```", 1)[0]
                data = json.loads(cleaned)
                st.session_state["extracted_data"] = data
                st.success(
                    f"Extraction successful! Confidence: {data.get('confidence', 'N/A')} | "
                    f"Document type: {data.get('document_type', 'N/A')}"
                )
            except json.JSONDecodeError:
                st.warning(
                    "AI response was not valid JSON. "
                    "Showing demo data as example."
                )
                _load_demo_extraction()
        else:
            st.info(
                "Showing demo data as fallback. "
                "Please check the error message above."
            )
            _load_demo_extraction()
    else:
        st.info(
            "Demo mode: Showing pre-built example extraction. "
            "For live extraction, enter an API key in the sidebar."
        )
        _load_demo_extraction()


def render_tab_extraction():
    st.header("PDF Extraction")
    st.markdown(
        "Upload any PDF or use the included sample datasheet. "
        "The app automatically extracts structured technical data."
    )

    uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")

    if uploaded is not None:
        try:
            _process_pdf_bytes(uploaded.read())
        except Exception:
            st.error("PDF processing failed.")
            _load_demo_extraction()

    else:
        if "extracted_data" not in st.session_state:
            st.info(
                "Upload your own PDF or test with the sample datasheet."
            )
            col_sample, col_demo = st.columns(2)
            with col_sample:
                if st.button(
                    "Use sample PDF",
                    key="load_sample_pdf",
                    type="primary",
                ):
                    if SAMPLE_PDF.exists():
                        try:
                            _process_pdf_bytes(SAMPLE_PDF.read_bytes())
                        except Exception:
                            st.error("Sample PDF could not be processed.")
                            _load_demo_extraction()
                    else:
                        st.warning("Sample PDF not found. Showing demo data.")
                        _load_demo_extraction()
            with col_demo:
                if st.button("Load demo data", key="load_demo_extraction"):
                    _load_demo_extraction()

    extracted = st.session_state.get("extracted_data")
    if extracted:
        st.subheader("Extracted Products")
        products = extracted.get("products", [])
        if products:
            df = pd.DataFrame(products)
            st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                key="extraction_editor",
            )

        if extracted.get("raw_requirements"):
            st.subheader("Detected Requirements")
            st.info(extracted["raw_requirements"])


def _load_demo_extraction():
    """Safely load demo extraction data into session state."""
    try:
        data = load_json(DEMO_EXTRACTION)
        if data:
            st.session_state["extracted_data"] = data
    except Exception:
        st.error("Demo data could not be loaded.")


# ---------------------------------------------------------------------------
# Tab 2: Semantic Search
# ---------------------------------------------------------------------------


def render_tab_search():
    st.header("Semantic Product Search")
    st.markdown(
        "Ask a natural language question and the app searches "
        "the product catalog semantically."
    )

    query = st.text_input(
        "Your query",
        placeholder="e.g. Which ventilator is suitable for an 80m2 server room with max 35 dB?",
        key="search_query",
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        max_sound = st.slider("Max. sound level (dB(A))", 20, 60, 50, key="filter_sound")
    with col_f2:
        min_airflow = st.slider("Min. airflow (m\u00b3/h)", 0, 200, 0, key="filter_airflow")
    with col_f3:
        filter_nfc = st.checkbox("NFC only", key="filter_nfc")
        filter_wrg = st.checkbox("HRV only", key="filter_wrg")
        filter_ex = st.checkbox("Ex-protection only", key="filter_ex")

    if query:
        collection, embed_model = build_chroma_collection()

        if collection is not None and embed_model is not None:
            with st.spinner("Searching product catalog..."):
                query_embedding = embed_model.encode([query]).tolist()
                results = collection.query(
                    query_embeddings=query_embedding, n_results=10
                )

            catalog = load_product_catalog()
            if catalog is None:
                st.error("Product catalog not available.")
                return
            products_by_id = {p["id"]: p for p in catalog["products"]}

            filtered = _filter_search_results(
                results, products_by_id, max_sound, min_airflow,
                filter_nfc, filter_wrg, filter_ex
            )

            api_key = get_api_key()
            if api_key and filtered:
                candidates_text = "\n".join(
                    f"- {r['product']['name']}: {r['product']['description']}"
                    for r in filtered
                )
                with st.spinner("Creating intelligent ranking..."):
                    system_prompt = read_prompt(PROMPT_MATCHING)
                    if system_prompt:
                        user_msg = (
                            f"Requirement: {query}\n\n"
                            f"Filters: max {max_sound} dB(A), min {min_airflow} m\u00b3/h\n\n"
                            f"Candidates:\n{candidates_text}"
                        )
                        ranking_text = call_claude(system_prompt, user_msg)

                        if ranking_text:
                            try:
                                cleaned = ranking_text.strip()
                                if cleaned.startswith("```"):
                                    cleaned = cleaned.split("\n", 1)[1]
                                    cleaned = cleaned.rsplit("```", 1)[0]
                                ranking_data = json.loads(cleaned)
                                st.session_state["search_results"] = ranking_data
                                _display_search_results(ranking_data)
                                return
                            except json.JSONDecodeError:
                                pass

            if filtered:
                st.info(
                    "For intelligent ranking, enter an API key. "
                    "Showing ChromaDB similarity scores."
                )
                for r in filtered:
                    p = r["product"]
                    with st.container():
                        st.markdown(
                            f"**#{r['rank']} {p['name']}** "
                            f"(Similarity: {r['score']:.2f})"
                        )
                        st.markdown(f"*{p['description']}*")
                        st.markdown(f"Use cases: {', '.join(p.get('use_cases', []))}")
                        st.divider()
            else:
                st.warning("No matching products found. Adjust your filters.")
        else:
            st.warning("Vector search not available. Showing demo results.")
            _load_demo_search()

    elif "search_results" in st.session_state:
        _display_search_results(st.session_state["search_results"])

    else:
        if st.button("Load demo search", key="load_demo_search"):
            _load_demo_search()


def _filter_search_results(
    results, products_by_id, max_sound, min_airflow,
    filter_nfc, filter_wrg, filter_ex
) -> list:
    """Filter ChromaDB results by user criteria."""
    filtered = []
    if not results or not results["ids"]:
        return filtered

    for i, pid in enumerate(results["ids"][0]):
        product = products_by_id.get(pid)
        if not product:
            continue
        specs = product.get("specs", {})

        sound = specs.get("sound_pressure_dba")
        if isinstance(sound, dict) and sound.get("low") is not None:
            if sound["low"] > max_sound:
                continue

        airflow = specs.get("airflow_m3h")
        max_af = 0
        if isinstance(airflow, list):
            max_af = max(airflow)
        elif isinstance(airflow, (int, float)):
            max_af = airflow
        if max_af < min_airflow and max_af > 0:
            continue

        if filter_nfc and not specs.get("nfc_configurable"):
            continue
        if filter_wrg and not specs.get("wrg_efficiency_pct"):
            continue
        if filter_ex and not specs.get("ex_rating"):
            continue

        score = 1 - (results["distances"][0][i] if results["distances"] else 0)
        filtered.append(
            {"product": product, "score": round(score, 3), "rank": len(filtered) + 1}
        )
        if len(filtered) >= 5:
            break

    return filtered


def _load_demo_search():
    """Safely load demo search data."""
    try:
        demo = load_json(DEMO_SEARCH)
        if demo:
            st.session_state["search_results"] = demo
            _display_search_results(demo)
    except Exception:
        st.error("Demo search data could not be loaded.")


def _display_search_results(data: dict):
    """Display ranked search results as cards."""
    if data.get("general_note"):
        st.info(data["general_note"])

    for rec in data.get("recommendations", []):
        with st.container():
            cols = st.columns([1, 6])
            with cols[0]:
                score = rec.get("score", 0)
                st.metric(f"#{rec['rank']}", f"{score:.0%}")
            with cols[1]:
                st.markdown(f"**{rec['product_name']}**")
                st.markdown(rec.get("reasoning", ""))
                if rec.get("caveats"):
                    for c in rec["caveats"]:
                        st.caption(f"Warning: {c}")
                meets = rec.get("meets_all_requirements", False)
                if meets:
                    st.success("Meets all requirements")
            st.divider()


# ---------------------------------------------------------------------------
# Tab 3: NFC Configuration
# ---------------------------------------------------------------------------


def render_tab_nfc():
    st.header("NFC Configuration Simulator")
    st.info(
        "Simulated configuration based on publicly documented ELS NFC "
        "parameters. Demonstrates the concept of automated parameterization."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Parameters")

        model = st.selectbox("Model", ELS_NFC_PARAMS["models"], key="nfc_model")

        st.markdown("**Airflow Stages**")
        steps = ELS_NFC_PARAMS["airflow_steps_m3h"]
        defaults = ELS_NFC_PARAMS["factory_defaults_m3h"]

        stage_1 = st.select_slider(
            "Stage 1 (m\u00b3/h)", options=steps,
            value=defaults["stage_1"], key="nfc_s1"
        )
        stage_2 = st.select_slider(
            "Stage 2 (m\u00b3/h)", options=steps,
            value=defaults["stage_2"], key="nfc_s2"
        )
        stage_3 = st.select_slider(
            "Stage 3 (m\u00b3/h)", options=steps,
            value=defaults["stage_3"], key="nfc_s3"
        )

        if stage_1 >= stage_2 or stage_2 >= stage_3:
            st.warning("Stages should be ascending: Stage 1 < Stage 2 < Stage 3")

        use_s4 = st.checkbox("Enable stage 4", key="nfc_use_s4")
        stage_4 = None
        if use_s4:
            stage_4 = st.select_slider(
                "Stage 4 (m\u00b3/h)", options=steps, value=80, key="nfc_s4"
            )

        use_s5 = st.checkbox("Enable stage 5", key="nfc_use_s5")
        stage_5 = None
        if use_s5:
            stage_5 = st.select_slider(
                "Stage 5 (m\u00b3/h)", options=steps, value=100, key="nfc_s5"
            )

        basic_vent = st.select_slider(
            "Basic ventilation (m\u00b3/h)", options=steps,
            value=ELS_NFC_PARAMS["basic_ventilation_m3h"], key="nfc_grund"
        )

        st.markdown("**Timing**")
        delay = st.slider("Start delay (s)", 0, 120, 5, key="nfc_delay")
        runon = st.slider("Run-on time (min)", 0, 90, 15, key="nfc_runon")
        interval = st.slider("Interval time (h)", 0, 24, 2, key="nfc_interval")

        sensor_config = None
        sensor_opts = ELS_NFC_PARAMS["sensor_options"]
        if model in sensor_opts:
            st.markdown("**Sensor Parameters**")
            opt = sensor_opts[model]

            if opt["type"] == "humidity":
                sensor_config = {
                    "type": "Humidity",
                    "unit": opt["unit"],
                    "threshold_rh": st.slider(
                        "Humidity threshold (%rH)", 40, 90, 65, key="nfc_rh"
                    ),
                }
            elif opt["type"] == "presence":
                sensor_config = {
                    "type": "Presence",
                    "detection": opt["detection"],
                }
            elif opt["type"] == "voc":
                voc_mode = st.selectbox("VOC mode", opt["modes"], key="nfc_voc_mode")
                voc_thresh = st.slider(
                    "VOC threshold",
                    opt["threshold_range"][0],
                    opt["threshold_range"][1],
                    250, key="nfc_voc_thresh",
                )
                voc_max = st.slider(
                    "VOC maximum",
                    opt["max_value_range"][0],
                    opt["max_value_range"][1],
                    400, key="nfc_voc_max",
                )
                sensor_config = {
                    "type": "VOC",
                    "mode": voc_mode,
                    "threshold_voc": voc_thresh,
                    "max_voc": voc_max,
                    "max_airflow_m3h": stage_3,
                }
            elif opt["type"] == "co2":
                sensor_config = {
                    "type": "CO2",
                    "unit": opt["unit"],
                    "threshold_ppm": st.slider(
                        "CO2 threshold (ppm)", 400, 2000, 1000, step=50, key="nfc_co2"
                    ),
                }

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Load factory defaults", key="nfc_reset"):
                for k in ["nfc_s1", "nfc_s2", "nfc_s3", "nfc_delay", "nfc_runon", "nfc_interval"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        with col_b2:
            if st.button("Import from extraction", key="nfc_from_extract"):
                extracted = st.session_state.get("extracted_data")
                if extracted and extracted.get("products"):
                    product = extracted["products"][0]
                    name = product.get("product_name", "")
                    for m in ELS_NFC_PARAMS["models"]:
                        if m.lower() in name.lower():
                            st.session_state["nfc_model"] = m
                            break
                    airflow = product.get("airflow_m3h")
                    if isinstance(airflow, (int, float)) and airflow in steps:
                        st.session_state["nfc_s3"] = airflow
                    st.success(f"Data imported from extraction: {name}")
                    st.rerun()
                else:
                    st.warning("No extracted data available. Use the Extraction tab first.")

    config = {
        "_meta": {
            "generator": "DocAI Assistant Prototype",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
            "disclaimer": "Simulated configuration - not for production use",
        },
        "device": {
            "model": model,
            "article_number": ARTICLE_NUMBERS.get(model),
        },
        "airflow_config": {
            "stage_1_m3h": stage_1,
            "stage_2_m3h": stage_2,
            "stage_3_m3h": stage_3,
            "stage_4_m3h": stage_4,
            "stage_5_m3h": stage_5,
            "basic_ventilation_m3h": basic_vent,
            "interval_m3h": basic_vent,
        },
        "timing": {
            "start_delay_sec": delay,
            "runon_time_min": runon,
            "interval_time_h": interval,
        },
    }
    if sensor_config:
        config["sensor"] = sensor_config

    st.session_state["nfc_config"] = config

    with col_right:
        st.subheader("Generated Configuration")
        config_json = json.dumps(config, indent=2, ensure_ascii=False)
        st.code(config_json, language="json")

        st.download_button(
            label="Download JSON",
            data=config_json,
            file_name=f"nfc_config_{model.replace(' ', '_')}.json",
            mime="application/json",
            key="nfc_download",
        )

        st.markdown("---")
        st.subheader("Parameter Explanation")
        sensor_label = (
            f"with {sensor_opts[model]['type']} sensor"
            if model in sensor_opts
            else "Base model without sensor"
        )
        st.markdown(
            f"- **Model:** {model} - {sensor_label}\n"
            f"- **Stages 1-3:** Standard airflow stages ({stage_1}/{stage_2}/{stage_3} m\u00b3/h)\n"
            f"- **Basic ventilation:** {basic_vent} m\u00b3/h continuous\n"
            f"- **Start delay:** {delay}s until fan starts\n"
            f"- **Run-on time:** {runon} min after trigger off\n"
            f"- **Interval time:** Automatic ventilation burst every {interval}h"
        )


# ---------------------------------------------------------------------------
# Tab 4: Energy Estimation
# ---------------------------------------------------------------------------


def render_tab_energy():
    st.header("Energy Savings Estimation (HRV)")
    st.warning(
        "Synthetic model based on physical approximation formulas - "
        "serves as a concept demonstration, not as a planning basis."
    )

    model, feature_cols = train_energy_model()

    col1, col2 = st.columns(2)
    with col1:
        room_size = st.slider("Room size (m\u00b2)", 20, 300, 80, key="energy_room")
        ceiling_height = st.slider(
            "Ceiling height (m)", 2.4, 4.0, 2.7, step=0.1, key="energy_ceil"
        )
        air_changes = st.slider(
            "Air change rate (/h)", 0.5, 5.0, 2.0, step=0.5, key="energy_ach"
        )
        wrg_eff = st.slider("HRV efficiency (%)", 50, 95, 75, key="energy_wrg")
    with col2:
        hours = st.slider("Operating hours/day", 6, 24, 12, key="energy_hours")
        delta_t = st.slider(
            "Temperature diff. indoor/outdoor (K)", 10, 30, 20, key="energy_dt"
        )
        heating_days = st.slider("Heating days/year", 180, 240, 210, key="energy_hd")
        price_ct = st.slider("Electricity price (ct/kWh)", 20, 50, 30, key="energy_price")

    X_input = pd.DataFrame(
        [
            {
                "room_size_m2": room_size,
                "air_changes_per_h": air_changes,
                "ceiling_height_m": ceiling_height,
                "wrg_efficiency_pct": wrg_eff,
                "hours_per_day": hours,
                "delta_t_k": delta_t,
                "heating_days": heating_days,
            }
        ]
    )
    prediction = model.predict(X_input)[0]

    tree_preds = np.array([t.predict(X_input)[0] for t in model.estimators_])
    pred_std = tree_preds.std()

    st.session_state["energy_result"] = {
        "kwh": prediction,
        "euro": prediction * price_ct / 100,
        "price_ct": price_ct,
    }

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Annual savings", f"{prediction:,.0f} kWh")
    with col_m2:
        savings_eur = prediction * price_ct / 100
        st.metric("Cost savings", f"{savings_eur:,.0f} EUR/year")
    with col_m3:
        co2_kg = prediction * DE_CO2_EMISSION_FACTOR_KG_PER_KWH
        st.metric("CO2 avoided", f"{co2_kg:,.0f} kg/year")

    st.subheader("Monthly Breakdown")

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    month_weights = [0.15, 0.14, 0.12, 0.08, 0.03, 0.0, 0.0, 0.0, 0.03, 0.10, 0.14, 0.15]
    total_w = sum(month_weights)
    monthly_kwh = [prediction * w / total_w for w in month_weights]
    monthly_upper = [(prediction + pred_std) * w / total_w for w in month_weights]
    monthly_lower = [max(0, (prediction - pred_std) * w / total_w) for w in month_weights]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=month_names, y=monthly_kwh,
            name="Savings (kWh)", marker_color=ACCENT_COLOR,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_names, y=monthly_upper, mode="lines",
            line={"dash": "dash", "color": "rgba(37,99,235,0.3)"},
            name="Upper bound", showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_names, y=monthly_lower, mode="lines",
            line={"dash": "dash", "color": "rgba(37,99,235,0.3)"},
            name="Lower bound", fill="tonexty",
            fillcolor="rgba(37,99,235,0.08)", showlegend=True,
        )
    )
    fig.update_layout(
        xaxis_title="Month", yaxis_title="Savings (kWh)",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model Details (Feature Importance)"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": importances}
        ).sort_values("Importance", ascending=True)

        fig_fi = go.Figure(
            go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h", marker_color=ACCENT_COLOR,
            )
        )
        fig_fi.update_layout(
            title="RandomForest Feature Importance",
            xaxis_title="Importance", template="plotly_white", height=300,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption(
            f"Model: RandomForestRegressor (n_estimators=100, n_samples=60). "
            f"Prediction Std: +/-{pred_std:,.0f} kWh"
        )


# ---------------------------------------------------------------------------
# Tab 5: Model Evaluation
# ---------------------------------------------------------------------------


def render_tab_evaluation():
    st.header("Model Evaluation")
    st.markdown(
        "Systematic comparison of different LLMs on the same extraction task. "
        "Same input document, same goal - which model extracts better?"
    )

    try:
        comparison = load_json(DEMO_COMPARISON)
        if comparison is None:
            st.error("Comparison data could not be loaded.")
            return
    except Exception:
        st.error("Comparison data could not be loaded.")
        return

    with st.expander("Input text (same document for both models)", expanded=False):
        st.text_area("Text", comparison["input_text"], height=150, disabled=True)

    st.subheader("Comparison: Claude Sonnet 4.6 vs. Llama-3.3-70B")

    claude_r = comparison["claude_result"]
    llama_r = comparison["llama_result"]

    comparison_data = {
        "Criterion": [
            "Completeness (fields extracted)",
            "JSON validity",
            "Latency",
            "Cost/1K tokens",
            "Text quality",
            "Structure adherence",
        ],
        "Claude Sonnet 4.6": [
            f"{claude_r['fields_extracted']}/{claude_r['fields_total']}",
            "Valid" if claude_r["json_valid"] else "Invalid",
            f"{claude_r['latency_ms']}ms",
            f"~${claude_r['cost_per_1k_tokens']}",
            claude_r["text_quality"],
            claude_r["structure_adherence"],
        ],
        "Llama-3.3-70B (Groq)": [
            f"{llama_r['fields_extracted']}/{llama_r['fields_total']}",
            ("Needed correction" if llama_r["json_needed_correction"] else "Valid"),
            f"{llama_r['latency_ms']}ms",
            f"~${llama_r['cost_per_1k_tokens']}",
            llama_r["text_quality"],
            llama_r["structure_adherence"],
        ],
    }

    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)

    if llama_r.get("issues"):
        with st.expander("Llama-3.3-70B: Identified Issues"):
            for issue in llama_r["issues"]:
                st.markdown(f"- {issue}")

    st.subheader("Extraction Results in Detail")
    col_c, col_l = st.columns(2)
    with col_c:
        st.markdown("**Claude Sonnet 4.6**")
        st.json(claude_r["extraction"])
    with col_l:
        st.markdown("**Llama-3.3-70B**")
        st.json(llama_r["extraction"])

    st.subheader("Conclusion")
    summary = comparison["comparison_summary"]
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Best Accuracy", summary["winner_accuracy"])
    with col_s2:
        st.metric("Fastest Model", summary["winner_speed"])
    with col_s3:
        st.metric("Cheapest Model", summary["winner_cost"])

    st.info(summary["recommendation"])
    st.caption(summary["key_insight"])


# ---------------------------------------------------------------------------
# Tab 6: PDF Report
# ---------------------------------------------------------------------------


def render_tab_report():
    st.header("PDF Report Export")
    st.markdown(
        "Generates a structured PDF report with all results "
        "from the other tabs."
    )

    st.subheader("Available Data")
    data_status = {
        "Extraction": "extracted_data" in st.session_state,
        "Product Search": "search_results" in st.session_state,
        "NFC Configuration": "nfc_config" in st.session_state,
        "Energy Estimation": "energy_result" in st.session_state,
    }

    for name, available in data_status.items():
        if available:
            st.success(f"{name}: Data available")
        else:
            st.warning(f"{name}: No data - use the tab first")

    any_data = any(data_status.values())

    if any_data:
        try:
            pdf_bytes = build_pdf_report()
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"docai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="download_report",
            )
        except Exception:
            st.error("Report generation failed.")
    else:
        st.info(
            "Use the other tabs first to generate data. "
            "The report summarizes all available results."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="DocAI Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    st.markdown(
        "<h1 style='text-align: center;'>DocAI Assistant</h1>"
        "<p style='text-align: center; font-size: 1.1em;'>"
        "<strong>AI-powered document analysis for HVAC equipment</strong><br>"
        "<em>Upload PDFs, extract structured data, search products semantically, "
        "generate reports</em></p>",
        unsafe_allow_html=True,
    )

    with st.expander("About DocAI Assistant", expanded=True):
        st.markdown(
            "**DocAI Assistant** is a functional AI prototype that demonstrates how "
            "intelligent document processing can optimize workflows for HVAC equipment "
            "manufacturers.\n\n"
            "**The Problem:** Technical data exists in unstructured formats "
            "(PDFs, datasheets, customer inquiries) and must be evaluated manually.\n\n"
            "**The Solution:** DocAI Assistant automatically extracts technical parameters from "
            "any document, finds matching products via natural language search, "
            "generates NFC configurations, and estimates energy savings - all AI-powered.\n\n"
            "**Works with any product catalog** - swap out the included HVAC dataset "
            "with your own products.json to adapt to your domain."
        )
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown(
            "The app works immediately - **no setup, no API key needed.** "
            "Just click through the tabs:\n\n"
            "**1. Extraction** - Click *\"Use sample PDF\"* or upload your own PDF. "
            "With API key: live AI extraction. Without: pre-built demo data.\n\n"
            "**2. Product Search** - Click *\"Load demo search\"* or type a query like "
            "*\"Quiet ventilator for 25m2 office with air quality sensor\"*. "
            "The app searches the product catalog semantically.\n\n"
            "**3. NFC Configuration** - Select a model and move the sliders. "
            "A JSON configuration is generated live.\n\n"
            "**4. Energy Estimation** - Set room parameters. "
            "The ML model calculates estimated annual savings in kWh, EUR, and CO2.\n\n"
            "**5. Model Evaluation** - Comparison: Claude Sonnet 4.6 vs. Llama-3.3-70B. "
            "Same document, same goal - which model extracts better?\n\n"
            "**6. PDF Report** - Combines all results into a downloadable PDF."
        )

    tabs = st.tabs(
        [
            "PDF Extraction",
            "Product Search",
            "NFC Configuration",
            "Energy Estimation",
            "Model Evaluation",
            "PDF Report",
        ]
    )

    with tabs[0]:
        render_tab_extraction()
    with tabs[1]:
        render_tab_search()
    with tabs[2]:
        render_tab_nfc()
    with tabs[3]:
        render_tab_energy()
    with tabs[4]:
        render_tab_evaluation()
    with tabs[5]:
        render_tab_report()


if __name__ == "__main__":
    main()
