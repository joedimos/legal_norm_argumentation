from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("vle_http_requests_total", "HTTP requests processed", ("method", "route", "status"))
DURATION = Histogram(
    "vle_http_request_duration_seconds",
    "HTTP request duration",
    ("method", "route"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
IN_FLIGHT = Gauge("vle_http_requests_in_flight", "HTTP requests currently being processed")
LIFECYCLE_RUNS = Counter("vle_lifecycle_runs_total", "Lifecycle evaluations", ("authorized", "high_stakes"))
