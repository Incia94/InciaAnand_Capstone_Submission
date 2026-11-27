"""Cirrus AI support workspace, in Streamlit: a lightweight way to run the supervisor and agent screens.

    streamlit run streamlit_app.py          # then open http://localhost:8501

Supervisor (e.g. marcus): every ticket from every channel, assigning them to agents, and Reports
(ticket volume, tickets by channel, how tickets were handled, and the decision log).
Agent (e.g. asha, sofia, ravi): their own tickets; each opens with what Cirrus found, the help
articles, and a suggested reply with references, which they check, edit if needed, and send.

It uses the same support hub, ticket store and decision log as the API (python -m src.api), so both
can run at once and show the same tickets. Demo accounts are created on first use (src/staff.py).
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cirrus AI · Support workspace", page_icon="☁️", layout="wide")


# ---------------------------------------------------------------- shared services (one per server)
@st.cache_resource(show_spinner="Starting Cirrus…")
def services():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from src.api import SupportPipeline
    from src.hub import SupportHub
    from src.staff import StaffDirectory
    staff = StaffDirectory()
    staff.ensure_demo()
    return SupportHub(SupportPipeline.from_env()), staff


hub, staff = services()
from src import workspace as W                      # noqa: E402  (after set_page_config)
from src.review import ReviewError                  # noqa: E402

user = st.session_state.get("user")


def header():
    left, right = st.columns([4, 1])
    left.markdown("## Cirrus AI")
    left.caption("Powered by CloudServe Solutions · Support workspace")
    if user:
        right.markdown(f"**{user['name']}**  \n{user['team']} · {user['role'].capitalize()}")
        if right.button("Sign out", width="stretch"):
            st.session_state.clear()
            st.rerun()
    st.divider()


def bar(rows: list[dict], x: str, y: str, horizontal: bool = False):
    if not rows:
        st.caption("No tickets yet.")
        return
    df = pd.DataFrame(rows)
    if horizontal:
        chart = alt.Chart(df).mark_bar().encode(y=alt.Y(f"{x}:N", sort="-x", title=None),
                                                x=alt.X(f"{y}:Q", title="Tickets"), tooltip=[x, y])
    else:
        chart = alt.Chart(df).mark_bar().encode(x=alt.X(f"{x}:N", sort=None, title=None),
                                                y=alt.Y(f"{y}:Q", title="Tickets"), tooltip=[x, y])
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------- sign-in
def sign_in():
    header()
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid, st.form("sign_in"):
        st.subheader("Staff sign-in")
        name = st.text_input("Username", placeholder="e.g. marcus or asha")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary", width="stretch"):
            found = staff.check(name, password)
            if found is None:
                st.error("Wrong username or password.")
            else:
                st.session_state["user"] = found
                st.rerun()
        st.caption("Supervisor: **marcus** · Agents: **asha**, **sofia**, **ravi** · demo password: **password**")


# ---------------------------------------------------------------- supervisor
def supervisor():
    header()
    queue_tab, reports_tab = st.tabs(["Ticket queue", "Reports"])
    with queue_tab:
        top = st.columns([3, 1])
        which = top[0].radio("Show", ["Unassigned", "Assigned"], horizontal=True, label_visibility="collapsed")
        if top[1].button("Add sample tickets", width="stretch",
                         help="Simulated tickets from email, live chat, the forum and docs comments"):
            with st.spinner("Cirrus is reading the new tickets…"):
                n = len(hub.simulate(count=8))
            st.toast(f"{n} sample tickets arrived.")
            st.rerun()
        tickets = W.queue(hub, which.lower())
        if not tickets:
            st.info("No tickets here. Use **Add sample tickets** to bring some in." if which == "Unassigned"
                    else "No tickets assigned yet.")
        agents = staff.list("agent")
        labels = {f"{a['name']} · {a['team']}": a for a in agents}
        for t in tickets:
            with st.container(border=True):
                c = st.columns([0.9, 3.4, 1, 0.9, 1.2])
                c[0].markdown(f"**{t['ticket_id']}**")
                c[1].markdown(f"**{t['title']}**  \n{t['customer']} · {t['age']}")
                c[2].markdown(t["channel"])
                c[3].markdown({"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}[t["priority"]])
                c[4].markdown(t["handled"] if which == "Unassigned" else f"**{t['assignee']}**  \n{t['work_status']}")
                if which == "Unassigned":
                    a, b, d = st.columns([2, 1, 3])
                    pick = a.selectbox("Assign to", list(labels), key=f"pick_{t['ticket_id']}",
                                       label_visibility="collapsed")
                    if b.button("Assign", key=f"assign_{t['ticket_id']}", type="primary"):
                        try:
                            hub.assign(t["ticket_id"], labels[pick], user)
                            st.toast(f"{t['ticket_id']} assigned to {labels[pick]['name']}.")
                            st.rerun()
                        except (LookupError, ValueError) as exc:
                            st.error(str(exc))
                    with d.expander("View ticket"):
                        v = W.agent_view(hub, t["ticket_id"])
                        st.markdown(f"**Customer report**\n\n{v['customer_report']}")
                        st.markdown(f"**Cirrus:** {v['findings']}")
    with reports_tab:
        reports()


def reports():
    r = W.reports(hub)
    k = st.columns(5)
    k[0].metric("Tickets", r["total"])
    k[1].metric("Unassigned", r["work"]["unassigned"])
    k[2].metric("Assigned", r["work"]["assigned"])
    k[3].metric("Resolved", r["work"]["resolved"])
    k[4].metric("Replies prepared by Cirrus", r["work"]["replies_prepared"])
    st.markdown("#### Ticket volume (last 14 days)")
    bar(r["volume"], "day", "tickets")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Tickets by channel")
        bar(r["by_channel"], "channel", "tickets", horizontal=True)
    with b:
        st.markdown("#### How tickets were handled")
        bar(r["handled"], "how", "tickets", horizontal=True)

    log = r["decision_log"]
    st.markdown("#### Decision log")
    if log["every_ticket_logged"]:
        st.success(f"Every ticket is logged: {log['tickets']} tickets, {log['decisions']} decisions.")
    else:
        st.error(f"{len(log['missing'])} ticket(s) have no final decision logged: {', '.join(log['missing'])}")
    rows = log["tickets_log"]
    if not rows:
        return
    find = st.text_input("Filter", placeholder="Ticket number, topic, channel or agent")
    if find:
        rows = [x for x in rows if find.lower() in " ".join(str(v) for v in x.values()).lower()]
    st.dataframe(pd.DataFrame(rows).rename(columns={
        "ticket_id": "Ticket", "title": "Subject", "channel": "Channel", "decision": "Decision", "why": "Why",
        "at": "Time", "steps": "Steps", "assignee": "Agent"}), hide_index=True, width="stretch")
    pick = st.selectbox("See every step for a ticket", [x["ticket_id"] for x in rows])
    if pick:
        st.dataframe(pd.DataFrame(W.ticket_steps(hub, pick)).rename(columns={
            "step": "Step", "decision": "Decision", "why": "Why", "at": "Time"}),
            hide_index=True, width="stretch")


# ---------------------------------------------------------------- agent
def agent():
    header()
    order = {"Open": 0, "In progress": 0, "Resolved": 1}
    mine = sorted((W.card(t) for t in hub.tickets.assigned_to(user["email"])),
                  key=lambda c: (order.get(c["work_status"], 0), c["ticket_id"]))
    left, right = st.columns([1.1, 2.4])
    with left:
        st.markdown("#### My tickets")
        show = st.radio("Show", ["To do", "Resolved", "All"], horizontal=True, label_visibility="collapsed")
        find = st.text_input("Search", placeholder="Search my tickets", label_visibility="collapsed")
        if show == "To do":
            mine = [t for t in mine if t["work_status"] != "Resolved"]
        elif show == "Resolved":
            mine = [t for t in mine if t["work_status"] == "Resolved"]
        if find:
            mine = [t for t in mine if find.lower() in f"{t['ticket_id']} {t['title']} {t['customer']}".lower()]
        if not mine:
            st.info("No tickets here. Your supervisor assigns tickets to you.")
            return
        ids = [t["ticket_id"] for t in mine]
        names = {t["ticket_id"]: f"{t['ticket_id']} · {t['title']}  \n{t['channel']} · {t['priority']} · "
                                 f"{t['work_status']}" for t in mine}
        current = st.session_state.get("open_ticket")
        tid = st.radio("Tickets", ids, index=ids.index(current) if current in ids else 0,
                       format_func=lambda i: names[i], label_visibility="collapsed")
        st.session_state["open_ticket"] = tid
    with right:
        ticket(tid)


def ticket(tid: str):
    v = W.agent_view(hub, tid)
    st.markdown(f"### {v['ticket_id']} · {v['title']}")
    st.caption(f"{v['customer']}{' · ' + v['customer_id'] if v['customer_id'] else ''} · {v['channel']} · "
               f"{v['priority']} priority · {v['age']} · {v['work_status']}")
    with st.container(border=True):
        st.markdown("**Customer report**")
        st.write(v["customer_report"])
        if len(v["conversation"]) > 1:
            with st.expander(f"Conversation ({len(v['conversation'])} messages)"):
                for m in v["conversation"]:
                    st.markdown(f"**{m['from']}** · {m['channel']}")
                    st.write(m["text"])
    with st.container(border=True):
        st.markdown("**What Cirrus found**")
        st.write(v["findings"])
        for a in v["articles"]:
            st.markdown(f"📄 [{a['title']}]({a['url']}) · {a['doc_id']}")

    if v["sent_reply"] and not v["can_send"]:
        st.success("Reply sent. This ticket is resolved.")
        with st.container(border=True):
            st.markdown("**Reply sent to the customer**")
            st.write(v["sent_reply"])
    elif v["work_status"] == "Resolved":
        st.success("This ticket is resolved.")
    else:
        st.markdown("**Suggested reply**" if v["reply_prepared"] else "**Your reply**")
        if v["subject"]:
            st.caption(f"Subject: {v['subject']}")
        if not v["reply_prepared"] and not v["reply"]:
            st.info("Cirrus didn't prepare a reply for this one. Write your own below.")
        text = st.text_area("Reply", value=v["reply"], height=300, key=f"reply_{tid}",
                            label_visibility="collapsed")
        b = st.columns([1, 1, 1, 2])
        if b[0].button("Save reply", key=f"save_{tid}", width="stretch"):
            hub.save_reply(tid, text)
            st.toast("Reply saved.")
        if b[1].button("Mark resolved", key=f"resolve_{tid}", width="stretch",
                       help="Close the ticket without sending a reply (for example, solved on a call)"):
            try:
                hub.resolve_without_reply(tid, user, "")
                st.rerun()
            except ReviewError as exc:
                st.error(str(exc))
        if b[2].button("Send reply", key=f"send_{tid}", type="primary", width="stretch",
                       disabled=not v["can_send"]):
            if not text.strip():
                st.error("Write a reply first.")
            else:
                try:
                    res = hub.send(tid, user, text.strip())
                    st.session_state.pop(f"reply_{tid}", None)
                    st.toast("Reply sent" + (" (with your edits)." if res["edited"] else "."))
                    st.rerun()
                except ReviewError as exc:
                    st.error(f"Not sent: {exc}")
        if not v["can_send"]:
            b[3].caption("Sending is off for this ticket; you can still save a reply or mark it resolved.")
    with st.expander("Decision trail"):
        st.dataframe(pd.DataFrame(v["trail"]).rename(columns={
            "step": "Step", "decision": "Decision", "why": "Why", "at": "Time"}),
            hide_index=True, width="stretch")


# ---------------------------------------------------------------- which screen
if not user:
    sign_in()
elif user["role"] == "supervisor":
    supervisor()
else:
    agent()
