import os
from collections import Counter
from uuid import uuid4
import requests
import streamlit as st

st.set_page_config(page_title='Cirrus | CloudServe', page_icon='☁️', layout='wide')
API = os.getenv('CIRRUS_API_URL', 'http://127.0.0.1:8000')


def call(method, path, **kwargs):
    try:
        response = requests.request(method, API + path, timeout=180, **kwargs)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        st.error(f'Backend request failed: {exc}')
        return None


st.sidebar.title('☁️ CIRRUS')
st.sidebar.caption('CloudServe support intelligence')
page = st.sidebar.radio('Workspace', ['Customer / channel simulator', 'Unified inbox', 'Analytics'])
st.sidebar.warning('Local demo only. Replies are simulated; no real messages are sent.')
if page == 'Customer / channel simulator':
    st.title('How can we help?')
    with st.form('ticket'):
        channel = st.selectbox('Source', ['chat', 'email', 'docs_comment', 'forum'])
        subject = st.text_input('Subject')
        message = st.text_area('Your CloudServe question', height=120)
        submit = st.form_submit_button('Ask Cirrus', type='primary')
    if submit:
        with st.spinner('Processing through Cirrus…'):
            result = call('POST', f'/desk/ingest/{channel}', json={
                'subject': subject, 'body': message, 'event_id': uuid4().hex})
        if result:
            st.session_state['last'] = result
    result = st.session_state.get('last')
    if result:
        o = result['outcome']
        st.caption(f"Ticket {result['id']} · AI-generated response")
        if o.get('reply'):
            st.write(o['reply'])
            st.json(o.get('sources', []), expanded=False)
        else:
            st.info('This request requires human support. It is available in the demo inbox.')
elif page == 'Unified inbox':
    st.title('Unified support inbox')
    rows = call('GET', '/desk/tickets') or []
    channel = st.selectbox('Filter source', ['all', 'chat', 'email', 'docs_comment', 'forum'])
    status = st.selectbox('Filter status', ['all', 'answered', 'escalated', 'resolved'])
    rows = [r for r in rows if channel in ('all', r['channel']) and status in ('all', r['status'])]
    if not rows:
        st.info('No matching tickets. Submit a question from the simulator.')
    else:
        selected = st.selectbox('Ticket', rows, format_func=lambda r: f"{r['channel']} | {r['status']} | {r['subject'] or r['id']}")
        item = call('GET', '/desk/tickets/' + selected['id'])
        if item:
            o = item['outcome']
            st.subheader(item['subject'] or item['id'])
            cols = st.columns(3)
            cols[0].metric('Intent', o.get('intent') or 'Unknown')
            cols[1].metric('Urgency', o.get('urgency') or 'Unknown')
            cols[2].metric('Confidence', f"{o.get('confidence', 0):.0%}")
            st.write(o.get('reason'))
            for msg in item['messages']:
                with st.chat_message('user' if msg['role'] == 'customer' else 'assistant'):
                    st.caption(msg['role'])
                    st.write(msg['body'])
            with st.expander('Sources and human handoff'):
                st.json({'sources': o.get('sources'), 'handoff': o.get('handoff'), 'draft': o.get('draft_reply')})
            draft = o.get('draft_reply') or {}
            text = st.text_area('Agent-reviewed reply (simulated)', value=draft.get('text', '') if draft.get('guardrail_passed') else '', key=item['id'])
            if st.button('Record agent reply') and text.strip():
                if call('POST', f"/desk/tickets/{item['id']}/reply", json={'body': text}):
                    st.rerun()
            if st.button('Mark resolved (agent confirmation)'):
                if call('PATCH', f"/desk/tickets/{item['id']}/status", json={'status': 'resolved'}):
                    st.rerun()
else:
    st.title('Support analytics')
    rows = call('GET', '/desk/tickets') or []
    c = st.columns(4)
    c[0].metric('Tickets', len(rows))
    c[1].metric('AI answered', sum(r['outcome']['outcome'] == 'sent' for r in rows))
    c[2].metric('Awaiting human', sum(r['status'] == 'escalated' for r in rows))
    c[3].metric('Agent-confirmed resolved', sum(r['status'] == 'resolved' for r in rows))
    if rows:
        st.bar_chart(dict(Counter(r['channel'] for r in rows)))
    st.caption('Live local inbox counts, not measured production CSAT or SLA.')
