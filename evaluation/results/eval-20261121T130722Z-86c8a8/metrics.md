# Evaluation report: eval-20261121T130722Z-86c8a8

Input `data/validation_tickets.json` (80 tickets, sha256 `94c5b1adc1203f06`), run 2026-11-21T13:07:22+00:00 to 2026-11-21T13:08:02+00:00 (39.7 s). Decision log reconciled: **yes**.

## Against the Brief's targets

| Target | Value | Goal | Met |
|---|---|---|---|
| Business: first contact resolution 60% or better | 78.8% | >= 60.0% | yes |
| Business: escalation rate 30% or lower | 21.2% | <= 30.0% | yes |
| Business: time to first reply under 5 minutes | 0.408 s | <= 300.000 s | yes |
| Technical: intent classification precision 85% or better | 100.0% | >= 85.0% | yes |
| Technical: hallucination rate 5% or lower | 0.0% | <= 5.0% | yes |
| Technical: citation accuracy 95% or better | 68.8% | >= 95.0% | no |
| Technical: response latency, 95th percentile, under 3 seconds | 0.814 s | < 3.000 s | yes |
| Governance: zero private data in outbound responses | 0 | == 0 | yes |
| Governance: under 5 points between customer groups | 22.2% | < 5.0% | no |
| Governance: stated confidence within 5 points of accuracy | 1.0% | <= 5.0% | yes |
| Governance: every automated decision logged | yes | == True | yes |

## Volume

| | |
|---|---|
| tickets processed | 80 |
| answered automatically | 63 |
| escalated | 17 |
| blocked by guardrails | 0 |
| errors | 0 |
| unprocessable input | 0 |
| tickets with labels | 80 |
| escalated by | policy 14, generate 2, documentation 1 |

## Escalations: genuine or avoidable

| | |
|---|---|
| escalated tickets with labels | 17 |
| genuine (labelled escalate) | 14 |
| avoidable: findability (labelled auto_respond, documented) | 3 |
| avoidable share of escalations | 17.6% |
| by category | {'policy': 14, 'documentation_incomplete': 2, 'no_documentation': 1} |
| avoidable, by category | {'documentation_incomplete': 2, 'no_documentation': 1} |
| by suggested team | {'tier1': 8, 'security': 4, 'product': 3, 'compliance': 1, 'tier2': 1} |
| answered through intent-guided search | 9 |
| escalations with a cited draft for the agent | 6 (6 passed every guardrail) |

- genuine = labelled escalate; avoidable = labelled auto_respond with a documented answer.
- Languages: {'en': 80}; translated 0; replies in the customer's language 0.

## Business outcomes

| Measure | This run | Baseline (history) |
|---|---|---|
| First contact resolution | 78.8% | 46.2% |
| Correct automatic resolution | 53.8% | n/a |
| Escalation rate | 21.2% | 53.8% |
| Response time, sent replies (mean / median) | 0.408 s / 0.406 s | 441.5 min / 218.0 min (time to resolution) |
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
| routing agreement with labels | 0.738 |
| retrieval hit rate | 94.3% |
| retrieval hit at 1 | 94.3% |
| citation accuracy | 68.8% |
| citations resolve to corpus | 100.0% |
| hallucination rate drafts | 0.0% |
| hallucination rate sent | 0.0% |
| latency p50 s | 0.408 s |
| latency p95 s | 0.814 s |
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

Degraded tickets: none. Generators: {'llm': 63}. Retrieval: {'hybrid': 80}.

## Governance

| Measure | Value |
|---|---|
| decisions logged | 551 |
| decision logging complete | yes |
| replies checked by guardrails | 63 |
| private data detections | 0 |
| private data in sent replies | 0 |
| must not auto respond answered automatically | 0 |
| fairness max gap | 22.2% |
| guardrail activations by type | none |

### Fairness by customer group

**customer_tier**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| business | 30 | 90.0% | 63.3% | 10.0% |
| enterprise | 8 | 87.5% | 75.0% | 12.5% |
| standard | 42 | 69.0% | 42.9% | 30.9% |

auto rate gap 20.9% (over 5 points); correct automatic resolution gap 20.5% (over 5 points); escalation rate gap 20.9% (over 5 points)

**customer_region**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| asia_pacific | 21 | 85.7% | 61.9% | 14.3% |
| europe | 25 | 80.0% | 56.0% | 20.0% |
| latin_america | 7 | 57.1% | 42.9% | 42.9% |
| north_america | 27 | 77.8% | 48.1% | 22.2% |

auto rate gap 7.9% (over 5 points); correct automatic resolution gap 13.8% (over 5 points); escalation rate gap 7.9% (over 5 points)

**language_fluency**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| fluent | 61 | 77.0% | 59.0% | 22.9% |
| non_fluent | 19 | 84.2% | 36.8% | 15.8% |

auto rate gap 7.2% (over 5 points); correct automatic resolution gap 22.2% (over 5 points); escalation rate gap 7.2% (over 5 points)

**channel**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| chat | 22 | 72.7% | 50.0% | 27.3% |
| docs_comment | 16 | 75.0% | 43.8% | 25.0% |
| email | 31 | 83.9% | 61.3% | 16.1% |
| forum | 11 | 81.8% | 54.5% | 18.2% |

auto rate gap 11.1% (over 5 points); correct automatic resolution gap 17.5% (over 5 points); escalation rate gap 11.1% (over 5 points)

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
    "prompt_version": "generate_answer.v2@070979e0+generate_answer_user.v1@43bfc0bd+repair_reply.v1@8fe5a390"
  },
  "kill_switch": false
}
```

The figures above are measurements. What they mean for CloudServe belongs in the report.
