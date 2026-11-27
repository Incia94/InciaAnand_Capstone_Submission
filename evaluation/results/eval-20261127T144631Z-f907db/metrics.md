# Evaluation report: eval-20261127T144631Z-f907db

Input `data/validation_tickets.json` (10 tickets, sha256 `94c5b1adc1203f06`), run 2026-11-27T14:46:31+00:00 to 2026-11-27T14:46:42+00:00 (10.9 s). Decision log reconciled: **yes**.

## Against the Brief's targets

| Target | Value | Goal | Met |
|---|---|---|---|
| Business: first contact resolution 60% or better | 30.0% | >= 60.0% | no |
| Business: escalation rate 30% or lower | 70.0% | <= 30.0% | no |
| Business: time to first reply under 5 minutes | 0.307 s | <= 300.000 s | yes |
| Technical: intent classification precision 85% or better | 100.0% | >= 85.0% | yes |
| Technical: hallucination rate 5% or lower | 0.0% | <= 5.0% | yes |
| Technical: citation accuracy 95% or better | 100.0% | >= 95.0% | yes |
| Technical: response latency, 95th percentile, under 3 seconds | 0.982 s | < 3.000 s | yes |
| Governance: zero private data in outbound responses | 0 | == 0 | yes |
| Governance: under 5 points between customer groups | n/a | < 5.0% | n/a |
| Governance: stated confidence within 5 points of accuracy | 1.0% | <= 5.0% | yes |
| Governance: every automated decision logged | yes | == True | yes |

## Volume

| | |
|---|---|
| tickets processed | 10 |
| answered automatically | 3 |
| conversational replies | 0 |
| escalated | 7 |
| blocked by guardrails | 0 |
| errors | 0 |
| unprocessable input | 0 |
| tickets with labels | 10 |
| escalated by | policy 4, channel_policy 3 |

## Escalations: genuine or avoidable

| | |
|---|---|
| escalated tickets with labels | 7 |
| genuine (labelled escalate) | 5 |
| avoidable: findability (labelled auto_respond, documented) | 0 |
| avoidable share of escalations | 0.0% |
| by category | {'policy': 4, 'channel_policy': 3} |
| avoidable, by category | none |
| by suggested team | {'tier1': 6, 'compliance': 1} |
| answered through intent-guided search | 1 |
| escalations with a cited draft for the agent | 4 (4 passed every guardrail) |

- genuine = labelled escalate; avoidable = labelled auto_respond with a documented answer.
- Languages: {'en': 10}; translated 0; replies in the customer's language 0.

## Business outcomes

| Measure | This run | Baseline (history) |
|---|---|---|
| First contact resolution | 30.0% | 40.0% |
| Correct automatic resolution | 30.0% | n/a |
| Escalation rate | 70.0% | 60.0% |
| Response time, sent replies (mean / median) | 0.307 s / 0.336 s | 537.5 min / 500.5 min (time to resolution) |
| Customer satisfaction | n/a (model-judged, with --judge) | 2.900 |

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
| urgency accuracy | 50.0% |
| calibration ece | 1.0% |
| calibration gap | 1.0% |
| routing agreement with labels | 0.800 |
| retrieval hit rate | 100.0% |
| retrieval hit at 1 | 100.0% |
| citation accuracy | 100.0% |
| citations resolve to corpus | 100.0% |
| hallucination rate drafts | 0.0% |
| hallucination rate sent | 0.0% |
| latency p50 s | 0.424 s |
| latency p95 s | 0.982 s |
| availability | 100.0% |

Per intent:

| Intent | Precision | Recall | Support |
|---|---|---|---|
| account_access | 100.0% | 100.0% | 1 |
| api_key_issue | 100.0% | 100.0% | 1 |
| api_usage_question | 100.0% | 100.0% | 1 |
| authentication_failure | 100.0% | 100.0% | 2 |
| billing_query | 100.0% | 100.0% | 1 |
| compliance_request | 100.0% | 100.0% | 1 |
| unclear_request | 100.0% | 100.0% | 3 |

Degraded tickets: none. Generators: {'extractive': 1, 'llm': 2}. Retrieval: {'hybrid': 10}.

## Governance

| Measure | Value |
|---|---|
| decisions logged | 86 |
| decision logging complete | yes |
| replies checked by guardrails | 3 |
| private data detections | 0 |
| private data in sent replies | 0 |
| must not auto respond answered automatically | 0 |
| fairness max gap | n/a |
| guardrail activations by type | none |

### Fairness audit (Governance Framework format)

Resolution rate = answered automatically. Quality score = correct automatic resolution (sent, right intent, an expected article cited, labelled auto_respond); with --judge, the mean model-judged rating is shown too. Variation from best = best group's quality score minus this group's. Groups under 10 tickets are listed but not used for the gap.

| Segment | Tickets in segment | Resolution rate | Quality score | Variation from best | Explanation |
|---|---|---|---|---|---|
| Enterprise customers | 0 | n/a | n/a | n/a | |
| Business customers | 4 | 25.0% | 25.0% | n/a | |
| Standard customers | 6 | 33.3% | 33.3% | n/a | |
| Tickets in fluent English | 8 | 37.5% | 37.5% | n/a | |
| Tickets in non-fluent English | 2 | 0.0% | 0.0% | n/a | |
| Short tickets (under 25 words) | 7 | 28.6% | 28.6% | n/a | |
| Long tickets (over 32 words) | 2 | 50.0% | 50.0% | n/a | |

The Explanation column is for your interpretation in the report.

### Fairness by customer group

**customer_tier**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| business | 4 | 25.0% | 25.0% | 75.0% |
| standard | 6 | 33.3% | 33.3% | 66.7% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

**customer_region**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| asia_pacific | 4 | 75.0% | 75.0% | 25.0% |
| europe | 3 | 0.0% | 0.0% | 100.0% |
| latin_america | 1 | 0.0% | 0.0% | 100.0% |
| north_america | 2 | 0.0% | 0.0% | 100.0% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

**language_fluency**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| fluent | 8 | 37.5% | 37.5% | 62.5% |
| non_fluent | 2 | 0.0% | 0.0% | 100.0% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

**channel**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| chat | 4 | 50.0% | 50.0% | 50.0% |
| docs_comment | 3 | 0.0% | 0.0% | 100.0% |
| email | 2 | 0.0% | 0.0% | 100.0% |
| forum | 1 | 100.0% | 100.0% | 0.0% |

auto rate gap n/a; correct automatic resolution gap n/a; escalation rate gap n/a

**ticket_length**

| Group | Tickets | Auto rate | Correct auto | Escalation |
|---|---|---|---|---|
| long (over 32 words) | 2 | 50.0% | 50.0% | 50.0% |
| medium (25 to 32 words) | 1 | 0.0% | 0.0% | 100.0% |
| short (under 25 words) | 7 | 28.6% | 28.6% | 71.4% |

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
    "model_calls_this_hour": 9,
    "hourly_call_budget": 600
  },
  "kill_switch": false
}
```

The figures above are measurements. What they mean for CloudServe belongs in the report.
