# Evaluation report: eval-20261127T185905Z-c3a210

Input `data/validation_tickets.json` (80 tickets, sha256 `94c5b1adc1203f06`), run 2026-11-27T18:59:05+00:00 to 2026-11-27T18:59:12+00:00 (7.2 s). Decision log reconciled: **yes**.

## Against the Brief's targets

| Target | Value | Goal | Met |
|---|---|---|---|
| Business: first contact resolution 60% or better | 56.2% | >= 60.0% | no |
| Business: escalation rate 30% or lower | 41.2% | <= 30.0% | no |
| Business: time to first reply under 5 minutes | 0.018 s | <= 300.000 s | yes |
| Technical: intent classification precision 85% or better | 95.7% | >= 85.0% | yes |
| Technical: hallucination rate 5% or lower | 0.0% | <= 5.0% | yes |
| Technical: citation accuracy 95% or better | 68.9% | >= 95.0% | no |
| Technical: response latency, 95th percentile, under 3 seconds | 0.020 s | < 3.000 s | yes |
| Governance: zero private data in outbound responses | 0 | == 0 | yes |
| Governance: under 5 points between customer groups | 29.0% | < 5.0% | no |
| Governance: stated confidence within 5 points of accuracy | 1.0% | <= 5.0% | yes |
| Governance: every automated decision logged | yes | == True | yes |

## Volume

| | |
|---|---|
| tickets processed | 80 |
| answered automatically | 45 |
| conversational replies | 2 |
| escalated | 33 |
| blocked by guardrails | 0 |
| errors | 0 |
| unprocessable input | 0 |
| tickets with labels | 80 |
| escalated by | channel_policy 20, policy 12, documentation 1 |

## Escalations: genuine or avoidable

| | |
|---|---|
| escalated tickets with labels | 33 |
| genuine (labelled escalate) | 18 |
| avoidable: findability (labelled auto_respond, documented) | 1 |
| avoidable share of escalations | 3.0% |
| by category | {'channel_policy': 20, 'policy': 12, 'no_documentation': 1} |
| avoidable, by category | {'no_documentation': 1} |
| by suggested team | {'tier1': 20, 'tier2': 5, 'security': 4, 'product': 3, 'compliance': 1} |
| answered through intent-guided search | 7 |
| escalations with a cited draft for the agent | 25 (24 passed every guardrail) |

- genuine = labelled escalate; avoidable = labelled auto_respond with a documented answer.
- Languages: {'en': 80}; translated 0; replies in the customer's language 0.

## Business outcomes

| Measure | This run | Baseline (history) |
|---|---|---|
| First contact resolution | 56.2% | 46.2% |
| Correct automatic resolution | 38.8% | n/a |
| Escalation rate | 41.2% | 53.8% |
| Response time, sent replies (mean / median) | 0.018 s / 0.016 s | 441.5 min / 218.0 min (time to resolution) |
| Customer satisfaction | n/a (model-judged, with --judge) | 2.600 |

- Brief definition: tickets closed without escalation / total tickets.
- Sent, right intent, an expected document cited, and labelled auto_respond. Labelled tickets only.
- System processing time from arrival to sent reply or escalation. An escalated ticket's time to a human reply is outside the system and not measured.
- Not measurable in a single run: needs follow-up tickets after a reply.

## Technical performance

| Measure | Value |
|---|---|
| intent accuracy | 97.5% |
| intent precision macro | 95.7% |
| intent recall macro | 94.2% |
| intent f1 macro | 94.8% |
| urgency accuracy | 46.2% |
| calibration ece | 1.0% |
| calibration gap | 1.0% |
| routing agreement with labels | 0.662 |
| retrieval hit rate | 94.3% |
| retrieval hit at 1 | 94.3% |
| citation accuracy | 68.9% |
| citations resolve to corpus | 100.0% |
| hallucination rate drafts | 0.0% |
| hallucination rate sent | 0.0% |
| latency p50 s | 0.016 s |
| latency p95 s | 0.020 s |
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
| unclassified | 0.0% | 0.0% | 0 |
| unclear_request | 100.0% | 66.7% | 6 |
| webhook_issue | 100.0% | 100.0% | 3 |

Degraded tickets: none. Generators: {'extractive': 31, 'llm': 14}. Retrieval: {'hybrid': 78}.

## Governance

| Measure | Value |
|---|---|
| decisions logged | 707 |
| decision logging complete | yes |
| replies checked by guardrails | 45 |
| private data detections | 0 |
| private data in sent replies | 0 |
| must not auto respond answered automatically | 0 |
| fairness max gap | 29.0% |
| guardrail activations by type | none |

### Fairness audit (Governance Framework format)

Resolution rate = answered automatically. Quality score = correct automatic resolution (sent, right intent, an expected article cited, labelled auto_respond); with --judge, the mean model-judged rating is shown too. Variation from best = best group's quality score minus this group's. Groups under 10 tickets are listed but not used for the gap.

| Segment | Tickets in segment | Resolution rate | Quality score | Variation from best | Explanation |
|---|---|---|---|---|---|
| Enterprise customers | 8 | 62.5% | 50.0% | 0.0% | |
| Business customers | 30 | 70.0% | 50.0% | 0.0% | |
| Standard customers | 42 | 45.2% | 28.6% | 21.4% | |
| Tickets in fluent English | 61 | 55.7% | 42.6% | 0.0% | |
| Tickets in non-fluent English | 19 | 57.9% | 26.3% | 16.3% | |
| Short tickets (under 25 words) | 32 | 65.6% | 46.9% | 0.0% | |
| Long tickets (over 32 words) | 22 | 45.5% | 31.8% | 15.1% | |

The Explanation column is for your interpretation in the report.

### Fairness by customer group

**customer_tier**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| business | 30 | 70.0% | 50.0% | 30.0% |
| enterprise | 8 | 62.5% | 50.0% | 37.5% |
| standard | 42 | 45.2% | 28.6% | 50.0% |

auto rate gap 24.8% (over 5 points); correct automatic resolution gap 21.4% (over 5 points); escalation rate gap 20.0% (over 5 points)

**customer_region**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| asia_pacific | 21 | 66.7% | 47.6% | 33.3% |
| europe | 25 | 52.0% | 36.0% | 48.0% |
| latin_america | 7 | 28.6% | 28.6% | 57.1% |
| north_america | 27 | 59.3% | 37.0% | 37.0% |

auto rate gap 14.7% (over 5 points); correct automatic resolution gap 11.6% (over 5 points); escalation rate gap 14.7% (over 5 points)

**language_fluency**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| fluent | 61 | 55.7% | 42.6% | 41.0% |
| non_fluent | 19 | 57.9% | 26.3% | 42.1% |

auto rate gap 2.1%; correct automatic resolution gap 16.3% (over 5 points); escalation rate gap 1.1%

**channel**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| chat | 22 | 72.7% | 50.0% | 22.7% |
| docs_comment | 16 | 43.8% | 25.0% | 50.0% |
| email | 31 | 51.6% | 38.7% | 48.4% |
| forum | 11 | 54.5% | 36.4% | 45.5% |

auto rate gap 29.0% (over 5 points); correct automatic resolution gap 25.0% (over 5 points); escalation rate gap 27.3% (over 5 points)

**ticket_length**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| long (over 32 words) | 22 | 45.5% | 31.8% | 54.5% |
| medium (25 to 32 words) | 26 | 53.8% | 34.6% | 46.2% |
| short (under 25 words) | 32 | 65.6% | 46.9% | 28.1% |

auto rate gap 20.2% (over 5 points); correct automatic resolution gap 15.1% (over 5 points); escalation rate gap 26.4% (over 5 points)

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
    "primary": "hybrid",
    "fallback_only": false,
    "reason": "",
    "strategy": "document",
    "thresholds": {
      "dense": 0.45,
      "lexical": 0.2273,
      "hybrid": 0.3731,
      "hybrid_hint": 0.2832,
      "lexical_hint": 0.094
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
    "prompt_version": "generate_answer.v3@57d47ef5+generate_answer_user.v1@43bfc0bd+repair_reply.v1@8fe5a390",
    "model_calls_this_hour": 47,
    "hourly_call_budget": 600
  },
  "kill_switch": false
}
```

The figures above are measurements. What they mean for CloudServe belongs in the report.
