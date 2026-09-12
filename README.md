<div align="center">

# Data Agent

### An AI-Powered Data Analysis and Cleaning Platform

Data Agent automatically profiles uploaded datasets, detects data quality issues, identifies the most likely prediction target, recommends suitable machine learning algorithms, generates visual insights, and produces AI-assisted cleaning recommendations — all from a single CSV upload.

[Live Demo](https://data-analyst-agent-topaz.vercel.app) &nbsp;•&nbsp; [Report a Bug](https://github.com/AaryaMakthala/DATA-AGENT/issues) &nbsp;•&nbsp; [Request a Feature](https://github.com/AaryaMakthala/DATA-AGENT/issues)

<br/>

> **Note:** The backend is hosted on Render's free tier, so the first request after a period of inactivity may take 2–3 minutes to respond. Thank you for your patience.

<br/>

![Next.js](https://img.shields.io/badge/Next.js%2015-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Workflow](#core-workflow)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Data Agent streamlines the traditional data analyst workflow. A user uploads a CSV file, and the system runs it through an automated pipeline that:

1. **Profiles** the dataset — row/column counts, data types, missing values, duplicates, and outliers
2. **Detects** columns that should be excluded from modeling — identifiers, GUIDs, and indexes
3. **Identifies** the most likely prediction target using a confidence-based scoring approach
4. **Classifies** the problem type as classification or regression
5. **Recommends** machine learning algorithms suited to the dataset's characteristics
6. **Generates** an AI-written analysis report explaining patterns, correlations, and data quality issues
7. **Produces** an interactive dashboard of charts and visualizations
8. **Applies** AI-recommended cleaning steps to produce a cleaned, model-ready dataset

The goal is to make rigorous, explainable data analysis accessible to anyone — without requiring deep data science expertise.

A core design principle behind the system is that **the LLM never sees raw data**. Only a compact statistical profile of the dataset is passed to the language model, minimizing cost, latency, and data exposure, while all actual data manipulation is performed deterministically in Python.

---

## System Architecture

```mermaid
flowchart TD
    classDef client fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef aiNode fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff;
    classDef pyNode fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef decision fill:#64748b,stroke:#475569,stroke-width:2px,color:#fff;
    classDef output fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;

    User(["User Uploads CSV"]):::client
    Frontend["Next.js Frontend"]:::client
    API["FastAPI Backend<br/>POST /upload"]:::client

    subgraph LG["LangGraph AI Pipeline"]
        direction TB
        Profiler["1. Dataset Profiler Node<br/><small>stats · types · missing values · outliers</small>"]:::pyNode
        TargetDetect["2. Target Detection Node<br/><small>runs on original, unencoded data</small>"]:::aiNode
        Validate{"3. Validation Node"}:::decision
        LLMAnalysis["4a. AI Dataset Analysis"]:::aiNode
        LLMClean["4b. AI Cleaning Strategy"]:::aiNode
        Cleaner["5. Python Cleaning Engine<br/><small>deterministic — no LLM</small>"]:::pyNode
        Viz["6. Visualization Node<br/><small>runs before one-hot encoding</small>"]:::pyNode
        MLRec["7. ML Recommendation Node"]:::pyNode
    end

    Invalid["Invalid Dataset Report"]:::output
    Report["Report JSON + Charts<br/>+ Cleaned CSV"]:::output
    Dashboard["Interactive Dashboard"]:::client

    User --> Frontend --> API --> Profiler
    Profiler --> TargetDetect --> Validate
    Validate -- "Invalid" --> Invalid --> Dashboard
    Validate -- "Valid" --> LLMAnalysis
    Validate -- "Valid" --> LLMClean
    LLMAnalysis --> Cleaner
    LLMClean --> Cleaner
    Cleaner --> Viz --> MLRec --> Report --> Dashboard
```

The LangGraph workflow executes the following node sequence:

```
START → Profiler Node → Target Detection Node → Validation Node
      → LLM Node (analysis + cleaning plan, run concurrently)
      → Python Cleaning Node → Visualization Node
      → ML Recommendation Node → END
```

If the validation node determines the dataset or detected target is unusable (for example, a single-class target), the workflow routes directly to `END` without invoking the LLM, cleaning, or visualization steps. The frontend then displays a clear invalid-dataset state instead of failing silently.

### Why This Architecture Holds Up

| Principle | How It's Achieved |
|---|---|
| **Privacy-first** | The LLM only ever receives a dataset profile, never the raw CSV |
| **Deterministic processing** | AI decides *what* to do; Python performs the *actual* cleaning |
| **Modular workflow** | LangGraph separates every stage into an independent, testable node |
| **Concurrent execution** | Dataset analysis and cleaning-plan generation run in parallel via a thread pool |
| **Graceful failure handling** | Validation catches invalid datasets early, surfacing meaningful error states |
| **Provider resilience** | Automatic fallback across Gemini → Groq → OpenRouter |
| **No database required** | Every upload is temporary; outputs are stored as files, keeping deployment simple |

---

## Core Workflow

```mermaid
flowchart TD
    A["Upload CSV Dataset"] --> B["Dataset Profiling"]
    B --> C["Target Detection<br/><small>on original, unencoded data</small>"]
    C --> D["Dataset Validation"]
    D -->|No| E["Invalid Dataset Report"]
    D -->|Yes| F["Parallel LLM Execution<br/><small>Dataset Analysis + Cleaning Strategy</small>"]
    F --> G["Python Cleaning Engine"]
    G --> H["Visualization<br/><small>before one-hot encoding</small>"]
    H --> I["ML Algorithm Recommendation"]
    I --> J["Report + Charts + Cleaned CSV"]
    J --> K["Interactive Dashboard"]

    style E fill:#ef4444,stroke:#b91c1c,color:#fff
    style K fill:#f59e0b,stroke:#b45309,color:#fff
```

Two design decisions are worth calling out:

- **Target detection runs before preprocessing.** Encoding a categorical target (e.g. turning `Purchased` into `Purchased_Yes` / `Purchased_No`) destroys the semantic information needed to identify it, so detection always runs against the original dataframe.
- **Visualizations are generated before one-hot encoding.** Encoded categorical columns produce unreadable, fragmented charts, so all charting happens on the cleaned-but-not-yet-encoded data.

---

## Key Features

### Automated Dataset Profiling
Analyzes uploaded CSV files and reports row/column statistics, data types, missing values, duplicates, outliers, unique value counts, and statistical summaries — all without sending the raw file to any LLM.

### Intelligent Target Detection
Rather than assuming the last column is the prediction target, the system uses a confidence-based scoring approach that evaluates column name semantics, business metric patterns, data type, cardinality, class imbalance, and general feature characteristics. Alternative target candidates are surfaced alongside their own confidence scores.

### Identifier Detection and Filtering
Columns that should never participate in machine learning — customer IDs, employee IDs, transaction IDs, GUIDs, and record indexes — are automatically detected and excluded from feature reasoning, correlation analysis, and chart generation.

### Machine Learning Algorithm Recommendation Engine
Algorithms are recommended based on actual dataset characteristics — size, feature composition, categorical ratio, outlier presence, class imbalance, and structure — rather than a static ranking.

| Problem Type | Recommended Algorithms |
|---|---|
| Classification | Gradient Boosting, Random Forest, XGBoost, Logistic Regression, Support Vector Machine |
| Regression | Gradient Boosting Regressor, Random Forest Regressor, XGBoost Regressor, Linear Regression |

Each recommendation includes a confidence score and a plain-language explanation of why the model fits the dataset.

### Data Quality and Cleaning Recommendations
Detects missing values, outliers, and duplicate records, and generates a cleaning strategy covering imputation, outlier treatment, categorical encoding, and column removal. The LLM only produces the *plan* — every transformation is executed deterministically by the Python cleaning engine, so results are reproducible and free of hallucination.

### AI-Powered Analysis Reports
An LLM generates natural-language insights describing patterns, correlations, data quality issues, and modeling considerations found in the dataset profile.

### Interactive Data Visualizations
Charts are generated automatically, including histograms, box plots, scatter plots, correlation heatmaps, count plots, bar charts, missing value charts, and target distribution charts.

### Resilient Multi-Provider LLM Execution
The two LLM calls in the pipeline — dataset analysis and cleaning plan generation — are independent and run concurrently via a thread pool rather than sequentially. If the primary provider fails, the system automatically falls back to the next configured provider with no user intervention required.

### ML Validation and Testing
Includes verification tests covering multiple dataset types (regression, classification with class imbalance, and invalid or single-class datasets) to confirm the pipeline correctly classifies problem type and rejects unusable data.

---

## Technology Stack

<table>
<tr>
<td valign="top" width="50%">

**Frontend**

![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)
![Framer](https://img.shields.io/badge/Framer_Motion-0055FF?style=flat-square&logo=framer&logoColor=white)
![shadcn/ui](https://img.shields.io/badge/shadcn%2Fui-000000?style=flat-square&logo=shadcnui&logoColor=white)
![Zod](https://img.shields.io/badge/Zod-3E67B1?style=flat-square&logo=zod&logoColor=white)

</td>
<td valign="top" width="50%">

**Backend**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-2A9D8F?style=flat-square&logo=gunicorn&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

</td>
</tr>
<tr>
<td valign="top" width="50%">

**AI / Agent Orchestration**

![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat-square&logo=groq&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-000000?style=flat-square&logo=openai&logoColor=white)

</td>
<td valign="top" width="50%">

**Auth & Deployment**

![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?style=flat-square&logo=supabase&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat-square&logo=vercel&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white)

</td>
</tr>
</table>

| Technology | Role in the Project |
|---|---|
| Next.js 15 (App Router) | Frontend framework and routing |
| React | Interactive UI components |
| TypeScript | Type safety across the frontend |
| Tailwind CSS | Styling system |
| Framer Motion | Animations and pipeline loading states |
| shadcn/ui | Reusable, accessible UI primitives |
| React Hook Form + Zod | Form handling and schema validation |
| FastAPI | REST API layer |
| Uvicorn | ASGI server |
| Pydantic | Request/response validation and configuration |
| Pandas / NumPy | Dataset loading, profiling, and cleaning |
| scikit-learn | Machine learning utilities for algorithm recommendation |
| LangGraph | State-machine orchestration of the AI pipeline |
| Google Gemini | Primary LLM provider |
| Groq | First fallback LLM provider |
| OpenRouter | Second fallback LLM provider |
| Python `ThreadPoolExecutor` | Concurrent execution of independent LLM calls |
| Supabase | Authentication |
| Vercel | Frontend deployment |
| Render | Backend deployment |

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Node.js 18 or later
- npm or yarn
- API keys for at least one supported LLM provider (Groq, Gemini, or OpenRouter)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000` and the backend API at `http://localhost:8000` by default.

---

## Project Structure

```
data-agent/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph workflow and node definitions
│   │   ├── api/              # FastAPI route definitions
│   │   ├── services/         # File handling and CSV validation services
│   │   ├── tools/             # Profiler, cleaner, visualizer, ML recommender
│   │   ├── llm/                # LLM provider clients and fallback logic
│   │   ├── models/             # Pydantic schemas
│   │   └── utils/               # Logging and shared utilities
│   ├── tests/                   # Verification and validation tests
│   ├── test_fixtures/           # Sample datasets used in testing
│   ├── uploads/                  # Temporary uploaded files
│   ├── outputs/
│   │   ├── charts/               # Generated chart images
│   │   ├── reports/               # Generated JSON analysis reports
│   │   └── cleaned_files/          # Cleaned, model-ready datasets
│   └── requirements.txt
├── frontend/
│   ├── app/                        # Next.js pages and routes
│   ├── components/                  # React components and UI elements
│   ├── hooks/                        # Custom React hooks
│   ├── lib/                           # Client utilities and API helpers
│   └── types/                          # TypeScript type definitions
└── README.md
```

---

## Environment Variables

The backend requires the following environment variables, typically defined in a `.env` file within the `backend` directory:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for the Groq LLM provider (primary) |
| `GEMINI_API_KEY` | API key for the Gemini LLM provider (fallback) |
| `OPENROUTER_API_KEY` | API key for the OpenRouter LLM provider (secondary fallback) |

---

## Testing

Backend verification tests can be run from the `backend` directory:

```bash
python -m pytest tests/
```

Tests cover target detection, problem type classification, class imbalance handling, and rejection of invalid or unusable datasets.

---

## Roadmap

- [ ] Asynchronous background processing via a task queue (Celery, Dramatiq, or RQ)
- [ ] Caching of dataset profiles to avoid recomputing results for identical uploads
- [ ] Interactive Plotly-based charts in place of static PNG images
- [ ] User authentication and persistent analysis history
- [ ] Explainable AI via SHAP feature importance for recommended models
- [ ] Support for Excel, Parquet, JSON, and compressed archive formats
- [ ] Data quality trend reports across multiple uploads
- [ ] Dockerized backend for consistent deployment across platforms
- [ ] Expanded automated test coverage across the full pipeline
- [ ] Upload security hardening: virus scanning, size limits, and stricter file-ID validation

---

## License

This project is currently unlicensed. Add a license file if you intend to distribute or open-source this project.

---

<div align="center">

Built by [Aarya Makthala](https://github.com/AaryaMakthala)

</div>
