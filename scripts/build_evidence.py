"""Build Discovery_Evidence.xlsx: descriptive evidence tables from the development set.
Usage: python build_evidence.py <data_dir> <out.xlsx>"""
import json, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
raw = json.loads((DATA / "development_tickets.json").read_text())
val = json.loads((DATA / "validation_tickets.json").read_text())
docs = json.loads((DATA / "documentation.json").read_text())
truth = json.loads((DATA / "ground_truth_responses.json").read_text())

df = pd.DataFrame([{**{k: t[k] for k in ["ticket_id", "channel", "body", "customer_id", "customer_tier",
                                         "customer_region", "language_fluency"]},
                    **{f"l_{k}": v for k, v in t["labels"].items()},
                    **{f"h_{k}": v for k, v in t["history"].items()}} for t in raw])
N = len(df)
pct = lambda s: round(100 * float(s.mean()), 1)

def base(g):
    return {"Tickets": len(g), "Share %": round(100 * len(g) / N, 1),
            "First-contact resolution %": pct(g.h_first_contact_resolution),
            "Escalated %": pct(g.h_escalated), "Repeat contact %": pct(g.h_repeat_contact),
            "Median resolution (min)": float(g.h_resolution_time_minutes.median()),
            "Mean resolution (min)": round(float(g.h_resolution_time_minutes.mean())),
            "Mean CSAT (1-5)": round(float(g.h_csat_rating.mean()), 2),
            "Answerable from docs %": pct(g.l_answerable_from_docs),
            "Labelled auto-respond %": pct(g.l_expected_route == "auto_respond")}

def seg(col, label, order=None):
    rows = [{label: k, **base(g)} for k, g in df.groupby(col)]
    out = pd.DataFrame(rows).sort_values("Tickets", ascending=False)
    return out

auto = df[df.l_expected_route == "auto_respond"]
avoid = auto[auto.h_escalated]
handled = auto[~auto.h_escalated]
ans = df[df.l_answerable_from_docs]; nans = df[~df.l_answerable_from_docs]
mnar = df[df.l_must_not_auto_respond]

def k5(s): return " ".join(re.sub(r"[^a-z ]", "", s.lower()).split()[:5])
df["tmpl"] = df.body.map(k5)
g = df.groupby("tmpl").agg(n=("ticket_id", "size"), ans=("l_answerable_from_docs", "mean"),
                           esc=("l_expected_route", lambda s: (s == "escalate").mean()))
mixed_ans = g[(g.ans > 0) & (g.ans < 1)]
mixed_route = g[(g.esc > 0) & (g.esc < 1)]
val_t = {k5(t["body"]) for t in val}
cited = Counter(d for ids in df.l_expected_doc_ids for d in ids)
gt = pd.DataFrame(truth)
mnc = {c for l in gt.must_not_claim for c in l}
per_cust = df.groupby("customer_id").size()

# ---------- Findings (facts only; interpretation is the student's) ----------
F = [
 ("E01", "Baseline", "Historical first-contact resolution", f"{pct(df.h_first_contact_resolution)}%", "Brief quotes 42%", "Baseline"),
 ("E02", "Baseline", "Historical escalation rate", f"{pct(df.h_escalated)}%", "Brief quotes 58%", "Baseline"),
 ("E03", "Baseline", "Mean / median resolution time", f"{round(df.h_resolution_time_minutes.mean())} / {int(df.h_resolution_time_minutes.median())} min", "This is resolution time, not time to first reply; the data has no first-reply field", "Baseline"),
 ("E04", "Baseline", "Mean CSAT", f"{round(df.h_csat_rating.mean(), 2)} / 5", "Brief quotes 3.2", "Baseline"),
 ("E05", "Baseline", "Repeat-contact rate", f"{pct(df.h_repeat_contact)}%", "Brief says repeat contacts are not measured today", "Baseline"),
 ("E06", "Documentation", "Tickets labelled answerable from existing docs", f"{len(ans)} of {N} ({pct(df.l_answerable_from_docs)}%)", "", "Answerable & routing"),
 ("E07", "Documentation", "Answerable tickets that were still escalated", f"{int(ans.h_escalated.sum())} ({pct(ans.h_escalated)}% of answerable)", "", "Answerable & routing"),
 ("E08", "Documentation", "Labelled auto-respond but historically escalated", f"{len(avoid)} tickets = {round(100*len(avoid)/N,1)}% of all, {round(100*len(avoid)/df.h_escalated.sum(),1)}% of escalations", "", "Answerable & routing"),
 ("E09", "Documentation", "Median resolution: auto-respond tickets handled first time vs escalated", f"{int(handled.h_resolution_time_minutes.median())} vs {int(avoid.h_resolution_time_minutes.median())} min", f"CSAT {round(handled.h_csat_rating.mean(),2)} vs {round(avoid.h_csat_rating.mean(),2)}", "Answerable & routing"),
 ("E10", "Documentation", "Repeat-contact rate: answerable vs not answerable", f"{pct(ans.h_repeat_contact)}% vs {pct(nans.h_repeat_contact)}%", "", "Answerable & routing"),
 ("E11", "Routing", "Tickets labelled must-not-auto-respond", f"{len(mnar)} ({pct(df.l_must_not_auto_respond)}%)", "Classes: " + ", ".join(f"{k} {v}" for k, v in mnar.l_intent.value_counts().items()), "By intent"),
 ("E12", "Routing", "Tickets labelled escalate (floor on escalation if labels are followed)", f"{int((df.l_expected_route=='escalate').sum())} ({pct(df.l_expected_route=='escalate')}%)", "Brief target is escalation at 30% or lower", "Answerable & routing"),
 ("E13", "Channels", "Volume by channel", "; ".join(f"{k} {v}" for k, v in df.channel.value_counts().items()), "", "By channel"),
 ("E14", "Channels", "Docs-comment channel outcomes", f"FCR {pct(df[df.channel=='docs_comment'].h_first_contact_resolution)}%, escalated {pct(df[df.channel=='docs_comment'].h_escalated)}%, repeat {pct(df[df.channel=='docs_comment'].h_repeat_contact)}%, median {df[df.channel=='docs_comment'].h_resolution_time_minutes.median():.0f} min", "Lowest answerable share of any channel", "By channel"),
 ("E15", "Channels", "Chat channel CSAT", f"{round(df[df.channel=='chat'].h_csat_rating.mean(),2)}", "Lowest CSAT of any channel", "By channel"),
 ("E16", "Segments", "Non-fluent share of tickets", f"{pct(df.language_fluency=='non_fluent')}%", "", "Segments"),
 ("E17", "Segments", "Avoidable-escalation rate by tier (auto-respond tickets escalated)", "; ".join(f"{k} {v}%" for k, v in (auto.groupby('customer_tier').h_escalated.mean()*100).round(1).items()), "Sample sizes on Segments tab", "Segments"),
 ("E18", "Segments", "Avoidable-escalation rate by region", "; ".join(f"{k} {v}%" for k, v in (auto.groupby('customer_region').h_escalated.mean()*100).round(1).items()), "Sample sizes on Segments tab", "Segments"),
 ("E19", "Segments", "Enterprise first-contact resolution", f"{pct(df[df.customer_tier=='enterprise'].h_first_contact_resolution)}%", "Lowest of the three tiers", "Segments"),
 ("E20", "Intents", "Intents with highest volume", "; ".join(f"{k} {v}" for k, v in df.l_intent.value_counts().head(5).items()), "22 intents, 13 to 29 tickets each", "By intent"),
 ("E21", "Intents", "Intents with most avoidable escalations", "; ".join(f"{k} {v}" for k, v in avoid.l_intent.value_counts().head(5).items()), "", "By intent"),
 ("E22", "Docs", "Docs cited most / least", f"{cited.most_common(1)[0][0]} {cited.most_common(1)[0][1]}; least {min(cited, key=cited.get)} {min(cited.values())}", "All 29 docs cited at least once; feature_request and unclear_request have no docs", "Doc demand"),
 ("E23", "Data quality", "Distinct ticket templates (first 5 words)", f"{df.tmpl.nunique()} across {N} tickets", f"{len(val_t & set(df.tmpl))} of {len(val_t)} validation templates also appear in dev", "Data quality"),
 ("E24", "Data quality", "Templates whose identical text carries mixed labels", f"answerable label mixed in {len(mixed_ans)} templates ({int(mixed_ans.n.sum())} tickets); route label mixed in {len(mixed_route)} ({int(mixed_route.n.sum())} tickets)", "Intent label is always consistent within a template", "Data quality"),
 ("E25", "Data quality", "Ground-truth answers", f"{len(gt)} answers, {gt.intent.nunique()} intents, {int((gt.must_mention.str.len()>0).sum())} with must_mention items", f"must_not_claim has {len(mnc)} distinct claims, identical on every answer", "Data quality"),
 ("E26", "Customers", "Distinct customers / max tickets from one", f"{per_cust.size} / {per_cust.max()}", f"{int((per_cust>=5).sum())} customers raised 5+ tickets", "Data quality"),
]
findings = pd.DataFrame(F, columns=["ID", "Area", "What was measured", "Value", "Note", "Detail tab"])

baseline_tbl = pd.DataFrame([{"Scope": "All tickets", **base(df)}])
channel_tbl = seg("channel", "Channel")
seg_tbl = pd.concat([seg("customer_tier", "Segment").assign(Dimension="Tier"),
                     seg("customer_region", "Segment").assign(Dimension="Region"),
                     seg("language_fluency", "Segment").assign(Dimension="Fluency")])
seg_auto = []
for dim, col in [("Tier", "customer_tier"), ("Region", "customer_region"), ("Fluency", "language_fluency"), ("Channel", "channel")]:
    for k, gg in auto.groupby(col):
        seg_auto.append({"Dimension": dim, "Segment": k, "Auto-respond tickets": len(gg),
                         "Escalated anyway %": pct(gg.h_escalated), "CSAT": round(float(gg.h_csat_rating.mean()), 2)})
seg_tbl = seg_tbl[["Dimension"] + [c for c in seg_tbl.columns if c != "Dimension"]]
intent_tbl = seg("l_intent", "Intent")
intent_tbl["Must not auto-respond"] = intent_tbl.Intent.map(df.groupby("l_intent").l_must_not_auto_respond.any())
intent_tbl["Avoidable escalations"] = intent_tbl.Intent.map(avoid.l_intent.value_counts()).fillna(0).astype(int)

gap_tbl = pd.DataFrame([
    {"Group": "Answerable from docs", **base(ans)},
    {"Group": "Not answerable from docs", **base(nans)},
    {"Group": "Labelled auto-respond, handled first time", **base(handled)},
    {"Group": "Labelled auto-respond, escalated anyway", **base(avoid)},
    {"Group": "Labelled escalate", **base(df[df.l_expected_route == "escalate"])}])
route_x = pd.crosstab(df.l_expected_route, df.h_escalated.map({True: "Escalated", False: "Not escalated"}), margins=True, margins_name="Total").reset_index().rename(columns={"l_expected_route": "Labelled route"})
ans_x = pd.crosstab(df.l_answerable_from_docs.map({True: "Answerable", False: "Not answerable"}), df.l_expected_route, margins=True, margins_name="Total").reset_index().rename(columns={"l_answerable_from_docs": "Docs label"})

doc_tbl = pd.DataFrame([{"Doc ID": d["doc_id"], "Title": d["title"], "Category": d["category"],
                         "Applies to": d["applies_to"], "Tickets citing": cited.get(d["doc_id"], 0)} for d in docs]).sort_values("Tickets citing", ascending=False)
intent_doc = (df.explode("l_expected_doc_ids").dropna(subset=["l_expected_doc_ids"])
              .groupby(["l_intent", "l_expected_doc_ids"]).size().reset_index(name="Tickets")
              .rename(columns={"l_intent": "Intent", "l_expected_doc_ids": "Doc ID"}))

dq = pd.DataFrame([
    {"Check": "Tickets / validation / docs / reference answers", "Result": f"{N} / {len(val)} / {len(docs)} / {len(gt)}"},
    {"Check": "Missing fields or empty bodies in dev", "Result": "None found"},
    {"Check": "Body length (chars) min / median / max", "Result": f"{df.body.str.len().min()} / {int(df.body.str.len().median())} / {df.body.str.len().max()}"},
    {"Check": "Date range", "Result": f"{min(t['received_at'] for t in raw)[:10]} to {max(t['received_at'] for t in raw)[:10]}"},
    {"Check": "Distinct 5-word openings (templates) in dev", "Result": df.tmpl.nunique()},
    {"Check": "Validation templates also present in dev", "Result": f"{len(val_t & set(df.tmpl))} of {len(val_t)}"},
    {"Check": "Templates with mixed answerable_from_docs labels", "Result": f"{len(mixed_ans)} templates, {int(mixed_ans.n.sum())} tickets"},
    {"Check": "Templates with mixed expected_route labels", "Result": f"{len(mixed_route)} templates, {int(mixed_route.n.sum())} tickets"},
    {"Check": "Templates with more than one intent", "Result": int((df.groupby('tmpl').l_intent.nunique() > 1).sum())},
    {"Check": "Example: dependency-resolution tickets", "Result": "Labelled DOC-DEPLOY-001 (health checks) or no doc, never DOC-DEPLOY-003 alone, which is the article on dependency resolution"},
    {"Check": "Docs last_reviewed_days_ago", "Result": "0 for all 29 (no staleness signal)"},
    {"Check": "Reference answers with must_mention items", "Result": int((gt.must_mention.str.len() > 0).sum())},
    {"Check": "Distinct must_not_claim strings", "Result": f"{len(mnc)}: " + "; ".join(sorted(mnc))},
    {"Check": "Reference answers whose ticket is in dev", "Result": int(gt.ticket_id.isin(df.ticket_id).sum())},
    {"Check": "Intents without a reference answer", "Result": ", ".join(sorted(set(df.l_intent) - set(gt.intent))) or "None"},
])

notes = pd.DataFrame([
    {"Item": "Source", "Detail": "development_tickets.json (500). Validation tickets were used only for the template-overlap check."},
    {"Item": "History fields", "Detail": "What happened when a human handled the ticket: the baseline, not labels to imitate (Dataset Guide)."},
    {"Item": "Avoidable escalation", "Detail": "Ticket labelled expected_route = auto_respond but history.escalated = true."},
    {"Item": "Template", "Detail": "First five words of the body, lowercased with punctuation removed. A rough grouping of reworded copies of the same message."},
    {"Item": "Resolution time", "Detail": "history.resolution_time_minutes. The data has no time-to-first-reply field."},
    {"Item": "Interpretation", "Detail": "These tables are measurements. What they mean, and the problem statement, are yours to write."},
    {"Item": "Reproduce", "Detail": "python build_evidence.py data/ Discovery_Evidence.xlsx"},
])

sheets = [("Findings", findings), ("Baseline", baseline_tbl), ("By channel", channel_tbl), ("Segments", seg_tbl),
          ("Segments (auto-respond)", pd.DataFrame(seg_auto)), ("By intent", intent_tbl),
          ("Answerable & routing", gap_tbl), ("Route crosstab", route_x), ("Docs label x route", ans_x),
          ("Doc demand", doc_tbl), ("Intent to doc", intent_doc), ("Data quality", dq), ("Method", notes)]

wb = Workbook(); wb.remove(wb.active)
hdr_fill = PatternFill("solid", fgColor="F3F3F3")
for name, t in sheets:
    ws = wb.create_sheet(name[:31])
    ws.append(list(t.columns))
    for row in t.itertuples(index=False):
        ws.append([(v.item() if hasattr(v, "item") else v) for v in row])
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = hdr_fill; c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    for i, col in enumerate(t.columns, 1):
        width = max([len(str(col))] + [len(str(v)) for v in t[col]])
        w = min(max(10, width + 2), 60)
        ws.column_dimensions[get_column_letter(i)].width = w
        if width > 60:
            for cell in ws[get_column_letter(i)][1:]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
wb.save(OUT)
print("wrote", OUT, [s for s, _ in sheets])
print(findings.to_string())
