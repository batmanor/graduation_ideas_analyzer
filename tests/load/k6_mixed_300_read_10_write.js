import http from "k6/http";
import { check, sleep } from "k6";
import { Gauge, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const MAX_RSS_MB = Number(__ENV.MAX_RSS_MB || 500);

const appRssMb = new Gauge("app_rss_mb");
const addVectorMs = new Trend("app_vector_store_add_vector_ms");
const createPaperMs = new Trend("app_paper_service_create_paper_ms");
const embeddingMs = new Trend("app_embedding_embed_ms");
const validationMs = new Trend("app_validation_validate_idea_ms");

export const options = {
  scenarios: {
    readers: {
      executor: "constant-vus",
      vus: 300,
      duration: __ENV.DURATION || "5m",
      exec: "validateIdea",
    },
    writers: {
      executor: "constant-vus",
      vus: 10,
      duration: __ENV.DURATION || "5m",
      exec: "addPaper",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.02"],
    http_req_duration: ["p(95)<1500", "p(99)<3000"],
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
  recordTiming(snapshot, "vector_store.add_vector", addVectorMs);
  recordTiming(snapshot, "paper_service.create_paper", createPaperMs);
  recordTiming(snapshot, "embedding.embed", embeddingMs);
  recordTiming(snapshot, "validation.validate_idea", validationMs);
}

export function validateIdea() {
  const payload = JSON.stringify({
    title: "Concurrent validation idea",
    abstract: "A proposal for checking novelty with vector search.",
    keywords: "novelty, validation, embeddings",
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

export function addPaper() {
  const uniqueId = Number(`${__VU}${Date.now()}${__ITER}`.slice(-12));
  const payload = JSON.stringify({
    external_id: uniqueId,
    title: `Load test paper ${uniqueId}`,
    abstract: "A synthetic paper generated during write-load testing.",
    keywords: "load, write, synthetic",
  });

  const res = http.post(`${BASE_URL}/api/v1/papers/`, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(res, {
    "paper create accepted": (r) => r.status === 202,
  });

  if (__ITER % 5 === 0) {
    sampleMetrics();
  }

  sleep(5);
}
