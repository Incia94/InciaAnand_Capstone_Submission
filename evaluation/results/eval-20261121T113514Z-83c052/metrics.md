# Evaluation report: eval-20261121T113514Z-83c052

Input `data/validation_tickets.json` (80 tickets, sha256 `94c5b1adc1203f06`), run 2026-11-21T11:35:14+00:00 to 2026-11-21T11:35:35+00:00 (21.3 s). Decision log reconciled: **yes**.

## Against the Brief's targets

| Target | Value | Goal | Met |
|---|---|---|---|
| Business: first contact resolution 60% or better | 50.0% | >= 60.0% | no |
| Business: escalation rate 30% or lower | 50.0% | <= 30.0% | no |
| Business: time to first reply under 5 minutes | 0.364 s | <= 300.000 s | yes |
| Technical: intent classification precision 85% or better | 100.0% | >= 85.0% | yes |
| Technical: hallucination rate 5% or lower | 0.0% | <= 5.0% | yes |
| Technical: citation accuracy 95% or better | 67.5% | >= 95.0% | no |
| Technical: response latency, 95th percentile, under 3 seconds | 0.538 s | < 3.000 s | yes |
| Governance: zero private data in outbound responses | 0 | == 0 | yes |
| Governance: under 5 points between customer groups | 31.8% | < 5.0% | no |
| Governance: stated confidence within 5 points of accuracy | 1.0% | <= 5.0% | yes |
| Governance: every automated decision logged | yes | == True | yes |

## Volume

| | |
|---|---|
| tickets processed | 80 |
| answered automatically | 40 |
| escalated | 40 |
| blocked by guardrails | 0 |
| errors | 0 |
| unprocessable input | 0 |
| tickets with labels | 80 |
| escalated by | documentation 24, policy 14, consistency 1, generate 1 |

## Business outcomes

| Measure | This run | Baseline (history) |
|---|---|---|
| First contact resolution | 50.0% | 46.2% |
| Correct automatic resolution | 33.8% | n/a |
| Escalation rate | 50.0% | 53.8% |
| Response time, sent replies (mean / median) | 0.364 s / 0.343 s | 441.5 min / 218.0 min (time to resolution) |
| Customer satisfaction | n/a (model-judged, with --judge) | 2.600 |

- Brief definition: tickets closed without escalation / total tickets.
- Sent, right intent, an expected document cited, and labelled auto_respond. Labelled tickets only.
- System processing time from arrival to sent reply or escalation. An escalated ticket's time to a human reply is outside the system and not measured.
- Not measurable in a single run: needs follow-up tickets after a reply.

## Technical performance

| Measure | Value |
|---|---|
| intent accuracy | 100.0% |
| intent precision macro | 100.0% |
| intent recall macro | 100.0% |
| intent f1 macro | 100.0% |
| urgency accuracy | 46.2% |
| calibration ece | 1.0% |
| calibration gap | 1.0% |
| routing agreement with labels | 0.600 |
| retrieval hit rate | 62.3% |
| retrieval hit at 1 | 60.4% |
| citation accuracy | 67.5% |
| citations resolve to corpus | 100.0% |
| hallucination rate drafts | 0.0% |
| hallucination rate sent | 0.0% |
| latency p50 s | 0.054 s |
| latency p95 s | 0.538 s |
| availability | 100.0% |

Per intent:

| Intent | Precision | Recall | Support |
|---|---|---|---|
| account_access | 100.0% | 100.0% | 4 |
| api_key_issue | 100.0% | 100.0% | 6 |
| api_usage_question | 100.0% | 100.0% | 4 |
| authentication_failure | 100.0% | 100.0% | 3 |
| billing_query | 100.0% | 100.0% | 10 |
| compliance_request | 100.0% | 100.0% | 1 |
| configuration_help | 100.0% | 100.0% | 3 |
| data_export | 100.0% | 100.0% | 1 |
| data_residency | 100.0% | 100.0% | 6 |
| database_issue | 100.0% | 100.0% | 1 |
| deployment_failure | 100.0% | 100.0% | 5 |
| feature_request | 100.0% | 100.0% | 3 |
| integration_help | 100.0% | 100.0% | 3 |
| onboarding | 100.0% | 100.0% | 4 |
| performance_degradation | 100.0% | 100.0% | 3 |
| quota_or_overage | 100.0% | 100.0% | 2 |
| rate_limit | 100.0% | 100.0% | 1 |
| rollback_request | 100.0% | 100.0% | 6 |
| security_incident | 100.0% | 100.0% | 4 |
| sso_configuration | 100.0% | 100.0% | 1 |
| unclear_request | 100.0% | 100.0% | 6 |
| webhook_issue | 100.0% | 100.0% | 3 |

Degraded tickets: none. Generators: {'llm': 40}. Retrieval: {'dense': 80}.

## Governance

| Measure | Value |
|---|---|
| decisions logged | 482 |
| decision logging complete | yes |
| replies checked by guardrails | 40 |
| private data detections | 0 |
| private data in sent replies | 0 |
| must not auto respond answered automatically | 0 |
| fairness max gap | 31.8% |
| guardrail activations by type | none |

### Fairness by customer group

**customer_tier**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| business | 30 | 56.7% | 43.3% | 43.3% |
| enterprise | 8 | 62.5% | 50.0% | 37.5% |
| standard | 42 | 42.9% | 23.8% | 57.1% |

auto rate gap 13.8% (over 5 points); correct automatic resolution gap 19.5% (over 5 points); escalation rate gap 13.8% (over 5 points)

**customer_region**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| asia_pacific | 21 | 47.6% | 38.1% | 52.4% |
| europe | 25 | 56.0% | 32.0% | 44.0% |
| latin_america | 7 | 28.6% | 28.6% | 71.4% |
| north_america | 27 | 51.8% | 33.3% | 48.1% |

auto rate gap 8.4% (over 5 points); correct automatic resolution gap 6.1% (over 5 points); escalation rate gap 8.4% (over 5 points)

**language_fluency**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| fluent | 61 | 49.2% | 34.4% | 50.8% |
| non_fluent | 19 | 52.6% | 31.6% | 47.4% |

auto rate gap 3.5%; correct automatic resolution gap 2.9%; escalation rate gap 3.5%

**channel**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| chat | 22 | 31.8% | 18.2% | 68.2% |
| docs_comment | 16 | 62.5% | 37.5% | 37.5% |
| email | 31 | 51.6% | 38.7% | 48.4% |
| forum | 11 | 63.6% | 45.5% | 36.4% |

auto rate gap 31.8% (over 5 points); correct automatic resolution gap 27.3% (over 5 points); escalation rate gap 31.8% (over 5 points)

## Run details

```
{
  "python": "3.11.18",
  "packages": {
    "langgraph": "1.2.24",
    "langchain-core": "1.8.4",
    "chromadb": "1.5.9",
    "scikit-learn": "1.7.2",
    "sentence-transformers": "6.3.0",
    "pydantic": "2.14.4"
  },
  "classifier": {
    "ready": true,
    "version": "tfidf-lr-C30-2e70ce0d"
  },
  "retrieval": {
    "primary": "dense",
    "fallback_only": false,
    "reason": "",
    "strategy": "document",
    "thresholds": {
      "dense": 0.5186,
      "lexical": 0.2273
    }
  },
  "routing": {
    "threshold": 0.99,
    "calibrated": true
  },
  "generator": {
    "mode": "llm",
    "client": "LangChainChatClient",
    "model": "meta-llama/llama-3.1-8b-instruct",
    "reason": "",
    "prompt_version": "generate_answer.v1@c23603fd+generate_answer_user.v1@43bfc0bd+repair_reply.v1@8fe5a390"
  },
  "kill_switch": false
}
```

The figures above are measurements. What they mean for CloudServe belongs in the report.
