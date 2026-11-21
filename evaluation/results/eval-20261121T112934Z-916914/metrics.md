# Evaluation report: eval-20261121T112934Z-916914

Input `data/validation_tickets.json` (5 tickets, sha256 `94c5b1adc1203f06`), run 2026-11-21T11:29:34+00:00 to 2026-11-21T11:29:48+00:00 (13.5 s). Decision log reconciled: **yes**.

## Against the Brief's targets

| Target | Value | Goal | Met |
|---|---|---|---|
| Business: first contact resolution 60% or better | 20.0% | >= 60.0% | no |
| Business: escalation rate 30% or lower | 80.0% | <= 30.0% | no |
| Business: time to first reply under 5 minutes | 1.511 s | <= 300.000 s | yes |
| Technical: intent classification precision 85% or better | 100.0% | >= 85.0% | yes |
| Technical: hallucination rate 5% or lower | 0.0% | <= 5.0% | yes |
| Technical: citation accuracy 95% or better | 100.0% | >= 95.0% | yes |
| Technical: response latency, 95th percentile, under 3 seconds | 1.236 s | < 3.000 s | yes |
| Governance: zero private data in outbound responses | 0 | == 0 | yes |
| Governance: under 5 points between customer groups | n/a | < 5.0% | n/a |
| Governance: stated confidence within 5 points of accuracy | 1.0% | <= 5.0% | yes |
| Governance: every automated decision logged | yes | == True | yes |

## Volume

| | |
|---|---|
| tickets processed | 5 |
| answered automatically | 1 |
| escalated | 4 |
| blocked by guardrails | 0 |
| errors | 0 |
| unprocessable input | 0 |
| tickets with labels | 5 |
| escalated by | policy 2, documentation 2 |

## Business outcomes

| Measure | This run | Baseline (history) |
|---|---|---|
| First contact resolution | 20.0% | 40.0% |
| Correct automatic resolution | 20.0% | n/a |
| Escalation rate | 80.0% | 60.0% |
| Response time, sent replies (mean / median) | 1.511 s / 1.511 s | 565.8 min / 543 min (time to resolution) |
| Customer satisfaction | n/a (model-judged, with --judge) | 2.400 |

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
| urgency accuracy | 80.0% |
| calibration ece | 1.0% |
| calibration gap | 1.0% |
| routing agreement with labels | 0.800 |
| retrieval hit rate | 50.0% |
| retrieval hit at 1 | 50.0% |
| citation accuracy | 100.0% |
| citations resolve to corpus | 100.0% |
| hallucination rate drafts | 0.0% |
| hallucination rate sent | 0.0% |
| latency p50 s | 0.032 s |
| latency p95 s | 1.236 s |
| availability | 100.0% |

Per intent:

| Intent | Precision | Recall | Support |
|---|---|---|---|
| api_key_issue | 100.0% | 100.0% | 1 |
| api_usage_question | 100.0% | 100.0% | 1 |
| authentication_failure | 100.0% | 100.0% | 1 |
| unclear_request | 100.0% | 100.0% | 2 |

Degraded tickets: none. Generators: {'llm': 1}. Retrieval: {'dense': 5}.

## Governance

| Measure | Value |
|---|---|
| decisions logged | 27 |
| decision logging complete | yes |
| replies checked by guardrails | 1 |
| private data detections | 0 |
| private data in sent replies | 0 |
| must not auto respond answered automatically | 0 |
| fairness max gap | n/a |
| guardrail activations by type | none |

### Fairness by customer group

**customer_tier**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| business | 2 | 50.0% | 50.0% | 50.0% |
| standard | 3 | 0.0% | 0.0% | 100.0% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

**customer_region**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| asia_pacific | 2 | 50.0% | 50.0% | 50.0% |
| europe | 1 | 0.0% | 0.0% | 100.0% |
| north_america | 2 | 0.0% | 0.0% | 100.0% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

**language_fluency**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| fluent | 3 | 33.3% | 33.3% | 66.7% |
| non_fluent | 2 | 0.0% | 0.0% | 100.0% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

**channel**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| chat | 2 | 0.0% | 0.0% | 100.0% |
| docs_comment | 1 | 0.0% | 0.0% | 100.0% |
| email | 1 | 0.0% | 0.0% | 100.0% |
| forum | 1 | 100.0% | 100.0% | 0.0% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

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
