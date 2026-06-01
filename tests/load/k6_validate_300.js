import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Gauge } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const MAX_RSS_MB = Number(__ENV.MAX_RSS_MB || "500");

const appRssMb = new Gauge("app_rss_mb");
const embeddingMs = new Trend("app_embedding_embed_ms");
const faissSearchMs = new Trend("app_faiss_search_ms");
const vectorSearchMs = new Trend("app_vector_store_search_ms");
const validationMs = new Trend("app_validation_validate_idea_ms");
const paperLookupMs = new Trend("app_paper_lookup_ms");

export const options = {
  scenarios: {
    validate_300_users: {
      executor: "constant-vus",
      vus: 300,
      duration: __ENV.DURATION || "5m",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<1000", "p(99)<2500"],
    app_rss_mb: [`value<${MAX_RSS_MB}`],
  },
};

function recordTiming(snapshot, name, metric) {
  const value = snapshot.timings?.[name]?.last_ms;
  if (typeof value === "number") {
    metric.add(value);
  }
}

function sampleMetrics() {
  const res = http.get(`${BASE_URL}/api/v1/metrics/`);
  check(res, {
    "metrics status is 200": (r) => r.status === 200,
  });

  if (res.status !== 200) {
    return;
  }

  const snapshot = res.json();
  if (typeof snapshot.process?.rss_mb === "number") {
    appRssMb.add(snapshot.process.rss_mb);
  }
  recordTiming(snapshot, "embedding.embed", embeddingMs);
  recordTiming(snapshot, "faiss.search", faissSearchMs);
  recordTiming(snapshot, "vector_store.search", vectorSearchMs);
  recordTiming(snapshot, "validation.validate_idea", validationMs);
  recordTiming(snapshot, "paper_service.get_papers_by_ids", paperLookupMs);
}

export default function () {
  const payload = JSON.stringify({
    title: "New retrieval idea",
    abstract: "A multilingual validation system for research ideas.",
    keywords: "retrieval, embeddings, faiss",
  });

  const res = http.post(`${BASE_URL}/api/v1/validate/`, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(res, {
    "validate status is 200": (r) => r.status === 200,
  });

  if (__ITER % 10 === 0) {
    sampleMetrics();
  }

  sleep(1);
}
