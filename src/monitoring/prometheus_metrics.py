"""Prometheus metric definitions for monitoring."""

from prometheus_client import Counter, Histogram, Gauge


# Request counters
REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total inference requests",
    ["model", "tier"],
)

# Token counters
TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["model", "direction"],  # direction: input | output
)

# Latency histograms
ROUTING_LATENCY = Histogram(
    "llm_routing_latency_seconds",
    "Router decision latency in seconds",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1],
)

INFERENCE_LATENCY = Histogram(
    "llm_inference_latency_seconds",
    "Model inference latency in seconds",
    ["model"],
)

TOTAL_LATENCY = Histogram(
    "llm_total_latency_seconds",
    "Total request latency in seconds",
)

# Cost histograms
COST_PER_REQUEST = Histogram(
    "llm_cost_per_request_usd",
    "Cost per request in USD",
    ["model"],
)

# Gauges
ROUTING_THRESHOLD = Gauge(
    "llm_routing_threshold",
    "Current routing complexity threshold",
)

COST_SAVINGS_PCT = Gauge(
    "llm_cost_savings_pct",
    "Rolling cost savings percentage vs always-large baseline",
)

MODEL_HEALTH = Gauge(
    "llm_model_health",
    "Model health status (1=healthy, 0=unhealthy)",
    ["model"],
)
