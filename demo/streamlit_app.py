"""Streamlit dashboard for the Cost-Aware LLM Routing System.

Usage:
    streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_settings
from src.config.model_registry import ModelRegistry
from src.cost.tracker import CostTracker
from src.inference.pipeline import InferencePipeline
from src.models.base import GenerationParams
from src.models.mock_backend import MockBackend


API_URL = os.getenv("API_URL", "").rstrip("/")
USE_API = bool(API_URL)


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def run_completion(prompt: str):
    """Run completion through FastAPI when API_URL is set, otherwise locally."""
    if USE_API:
        response = httpx.post(
            f"{API_URL}/v1/completions",
            json={"prompt": prompt},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return SimpleNamespace(
            text=data["text"],
            model_used=data["model_used"],
            model_tier=data["model_tier"],
            routing_decision=_to_namespace(data["routing"]),
            cost=_to_namespace(data["cost"]),
            total_latency_ms=data["latency_ms"],
        )

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(get_pipeline().run(prompt))
    loop.close()
    return result


def get_cost_summary():
    """Get analytics summary from FastAPI or the local pipeline."""
    if USE_API:
        response = httpx.get(f"{API_URL}/v1/analytics/costs", timeout=10)
        response.raise_for_status()
        return _to_namespace(response.json())
    return get_pipeline().cost_tracker.get_summary()


def reset_analytics():
    """Reset analytics through FastAPI or the local pipeline."""
    if USE_API:
        response = httpx.post(f"{API_URL}/v1/analytics/reset", timeout=10)
        response.raise_for_status()
    else:
        get_pipeline().cost_tracker.reset()


# --- Session State Initialization ---

def get_pipeline() -> InferencePipeline:
    """Get or create the pipeline singleton in session state."""
    if "pipeline" not in st.session_state:
        settings = load_settings(mode="simulation")
        registry = ModelRegistry()
        registry.load_from_yaml()

        backends = {
            settings.models.small.name: MockBackend(
                model_name=settings.models.small.name,
                tier="small",
                avg_latency_ms=settings.models.small.avg_latency_ms,
            ),
            settings.models.large.name: MockBackend(
                model_name=settings.models.large.name,
                tier="large",
                avg_latency_ms=settings.models.large.avg_latency_ms,
            ),
        }

        st.session_state.pipeline = InferencePipeline(
            settings=settings,
            model_registry=registry,
            backends=backends,
        )

    return st.session_state.pipeline


def init_history():
    """Initialize request history in session state."""
    if "history" not in st.session_state:
        st.session_state.history = []


# --- Page Config ---

st.set_page_config(
    page_title="LLM Router - Cost-Aware Routing",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Cost-Aware LLM Routing System")
st.caption("Intelligent query routing to minimize inference cost while maintaining quality")

# --- Sidebar ---

with st.sidebar:
    st.header("Settings")

    init_history()

    if USE_API:
        st.info(f"Using API: {API_URL}")
    else:
        pipeline = get_pipeline()
        threshold = st.slider(
            "Routing Threshold",
            min_value=0.0,
            max_value=1.0,
            value=pipeline.routing_engine.threshold_manager.threshold,
            step=0.05,
            help="Prompts with complexity score above this threshold are routed to the large model.",
        )
        pipeline.routing_engine.threshold_manager.threshold = threshold

        st.divider()

        mode = st.radio(
            "Threshold Mode",
            ["Custom", "Conservative (0.3)", "Default (0.5)", "Aggressive (0.7)"],
            index=0,
        )
        if mode.startswith("Conservative"):
            pipeline.routing_engine.threshold_manager.set_mode("conservative")
        elif mode.startswith("Default"):
            pipeline.routing_engine.threshold_manager.set_mode("default")
        elif mode.startswith("Aggressive"):
            pipeline.routing_engine.threshold_manager.set_mode("aggressive")

    st.divider()

    st.subheader("Model Info")
    settings = load_settings(mode="production" if USE_API else "simulation")
    st.markdown(f"""
    **Small Model:** {settings.models.small.name}
    - Cost: ${settings.models.small.cost_per_1k_tokens}/1K tokens
    - Latency: ~{settings.models.small.avg_latency_ms}ms

    **Large Model:** {settings.models.large.name}
    - Cost: ${settings.models.large.cost_per_1k_tokens}/1K tokens
    - Latency: ~{settings.models.large.avg_latency_ms}ms
    """)

    if st.button("Reset Analytics"):
        reset_analytics()
        st.session_state.history = []
        st.success("Analytics reset!")

# --- Tabs ---

tab1, tab2, tab3, tab4 = st.tabs(["Try It", "Analytics", "Batch Benchmark", "Architecture"])

# --- Tab 1: Interactive Testing ---

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            height=120,
            placeholder="Type a question or task here...",
        )

        # Quick examples
        st.caption("Quick examples:")
        example_cols = st.columns(3)
        with example_cols[0]:
            if st.button("Simple: What is Python?", use_container_width=True):
                prompt = "What is Python?"
        with example_cols[1]:
            if st.button("Medium: Explain TCP/IP", use_container_width=True):
                prompt = "Explain how TCP/IP networking works step by step"
        with example_cols[2]:
            if st.button("Complex: Implement BST", use_container_width=True):
                prompt = "Write a Python implementation of a balanced binary search tree with insertion, deletion, and search. Analyze time complexity."

        if st.button("Route & Generate", type="primary", use_container_width=True) and prompt:
            with st.spinner("Routing and generating..."):
                result = run_completion(prompt)

            # Store in history
            st.session_state.history.append({
                "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                "model": result.model_used,
                "tier": result.model_tier,
                "score": result.routing_decision.complexity_score,
                "cost": result.cost.cost_usd,
                "savings_pct": result.cost.savings_pct,
                "latency_ms": result.total_latency_ms,
            })

            # Display results
            st.subheader("Generated Response")
            st.markdown(result.text)

    with col2:
        if prompt and st.session_state.history:
            last = st.session_state.history[-1]

            st.subheader("Routing Decision")

            # Model badge
            if last["tier"] == "small":
                st.success(f"Routed to: **{last['model']}** (small)")
            else:
                st.warning(f"Routed to: **{last['model']}** (large)")

            # Complexity gauge
            st.metric("Complexity Score", f"{last['score']:.2f}")

            # Cost info
            st.subheader("Cost Breakdown")
            st.metric("Request Cost", f"${last['cost']:.6f}")
            st.metric("Savings vs Large Model", f"{last['savings_pct']:.1f}%")
            st.metric("Latency", f"{last['latency_ms']:.1f}ms")

            # Routing explanation
            if "result" in dir():
                st.subheader("Why this route?")
                for reason in result.routing_decision.reasoning:
                    st.markdown(f"- {reason}")


# --- Tab 2: Analytics ---

with tab2:
    summary = get_cost_summary()

    if summary.total_requests == 0:
        st.info("No requests yet. Try some prompts in the 'Try It' tab to see analytics.")
    else:
        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Requests", summary.total_requests)
        m2.metric("Total Cost", f"${summary.total_cost_usd:.4f}")
        m3.metric("Total Savings", f"${summary.total_savings_usd:.4f}")
        m4.metric("Savings %", f"{summary.savings_pct:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            # Routing distribution pie chart
            if summary.requests_by_model:
                fig = px.pie(
                    names=list(summary.requests_by_model.keys()),
                    values=list(summary.requests_by_model.values()),
                    title="Routing Distribution",
                    color_discrete_sequence=["#2ecc71", "#e74c3c"],
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cost by model bar chart
            if summary.cost_by_model:
                fig = px.bar(
                    x=list(summary.cost_by_model.keys()),
                    y=list(summary.cost_by_model.values()),
                    title="Cost by Model",
                    labels={"x": "Model", "y": "Cost (USD)"},
                    color=list(summary.cost_by_model.keys()),
                    color_discrete_sequence=["#2ecc71", "#e74c3c"],
                )
                st.plotly_chart(fig, use_container_width=True)

        # Cost comparison
        st.subheader("Cost Comparison")
        comparison_data = {
            "Scenario": ["With Routing", "Always Large Model"],
            "Total Cost (USD)": [summary.total_cost_usd, summary.total_cost_if_large_usd],
        }
        fig = px.bar(
            pd.DataFrame(comparison_data),
            x="Scenario",
            y="Total Cost (USD)",
            title="Cost: Routing vs Always Large",
            color="Scenario",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Request history table
        if st.session_state.history:
            st.subheader("Request History")
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)


# --- Tab 3: Batch Benchmark ---

with tab3:
    st.subheader("Batch Benchmark")
    st.markdown("Run the router against a batch of sample prompts to measure accuracy and cost savings.")

    sample_path = Path(__file__).parent / "sample_prompts.json"

    if st.button("Run Benchmark", type="primary"):
        with open(sample_path) as f:
            data = json.load(f)

        prompts = data["prompts"]
        results = []

        progress = st.progress(0)
        for i, item in enumerate(prompts):
            result = run_completion(item["text"])
            correct = result.model_tier == item["expected_tier"]
            results.append({
                "prompt": item["text"][:60] + "...",
                "expected": item["expected_tier"],
                "actual": result.model_tier,
                "correct": correct,
                "score": result.routing_decision.complexity_score,
                "cost": result.cost.cost_usd,
                "savings_pct": result.cost.savings_pct,
            })
            progress.progress((i + 1) / len(prompts))

        df = pd.DataFrame(results)
        accuracy = df["correct"].mean()

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Routing Accuracy", f"{accuracy:.0%}")
        m2.metric("Total Prompts", len(prompts))
        m3.metric("Total Cost", f"${df['cost'].sum():.4f}")
        m4.metric("Avg Savings", f"{df['savings_pct'].mean():.1f}%")

        # Confusion matrix
        col1, col2 = st.columns(2)
        with col1:
            # Accuracy by expected tier
            tier_acc = df.groupby("expected")["correct"].mean().reset_index()
            tier_acc.columns = ["Expected Tier", "Accuracy"]
            fig = px.bar(
                tier_acc,
                x="Expected Tier",
                y="Accuracy",
                title="Accuracy by Expected Tier",
                color="Expected Tier",
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Complexity score distribution
            fig = px.histogram(
                df,
                x="score",
                color="expected",
                nbins=20,
                title="Complexity Score Distribution",
                barmode="overlay",
                opacity=0.7,
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Full results table
        st.subheader("Detailed Results")
        st.dataframe(
            df.style.apply(
                lambda row: ["background-color: #d4edda" if row["correct"] else "background-color: #f8d7da"] * len(row),
                axis=1,
            ),
            use_container_width=True,
        )


# --- Tab 4: Architecture ---

with tab4:
    st.subheader("System Architecture")

    st.markdown("""
    ### Overview

    The Cost-Aware LLM Routing System intelligently routes queries to the most
    cost-effective model based on prompt complexity analysis.

    ### Pipeline Flow

    ```
    Client Request
         |
         v
    ┌─────────────────┐
    │   FastAPI API    │  POST /v1/completions
    └────────┬────────┘
             |
             v
    ┌─────────────────┐
    │  Preprocessor   │  Text cleaning, tokenization
    └────────┬────────┘
             |
             v
    ┌─────────────────┐
    │  Router Engine   │  DistilBERT classifier + rule-based features
    │  (< 10ms)       │  → Complexity score [0, 1]
    └────────┬────────┘
             |
        score < threshold?
        /              \\
       v                v
    ┌──────────┐  ┌──────────┐
    │  Small   │  │  Large   │
    │  Model   │  │  Model   │
    │ (Phi-2)  │  │(Llama-2) │
    │  $0.002  │  │  $0.06   │
    └────┬─────┘  └────┬─────┘
         |              |
         v              v
    ┌─────────────────┐
    │  Cost Tracker   │  Track cost, calculate savings
    └────────┬────────┘
             |
             v
    ┌─────────────────┐
    │  Response +     │  Text + routing info + cost breakdown
    │  Metadata       │
    └─────────────────┘
    ```

    ### Key Components

    | Component | Technology | Purpose |
    |-----------|-----------|---------|
    | Router | DistilBERT + Rules | Classify prompt complexity |
    | Optimization | TensorRT (FP16/INT8) | Accelerate model inference |
    | Serving | Triton Inference Server | Production model serving |
    | API | FastAPI | REST endpoints |
    | Monitoring | Prometheus | Metrics collection |
    | Demo | Streamlit | Interactive dashboard |

    ### Cost Savings Mechanism

    The router identifies that **many queries don't need the largest model**.
    Simple factual questions, greetings, and short tasks are handled by the
    small model at **1/30th the cost** while maintaining adequate quality.

    Only complex tasks (code generation, multi-step reasoning, mathematical
    proofs) are routed to the expensive large model.

    **Result: 50-85% cost savings** compared to always using the large model.
    """)
