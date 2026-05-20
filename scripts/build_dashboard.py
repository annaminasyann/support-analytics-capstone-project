"""
Interactive Dashboard — Support Analytics Chunk Pipeline
Run: python scripts/build_dashboard.py && open dashboard.html
"""

import sys, base64
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gcp_config import PROJECT_ID, bq_table
from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)

def load(sql):
    return client.query(sql).to_dataframe()

def try_load(sql, fallback_cols):
    try:
        return load(sql)
    except Exception as e:
        print(f'  [WARN] {e}')
        return pd.DataFrame(columns=fallback_cols)

def load_csv(name):
    p = ROOT / 'data' / 'results' / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def embed_png(rel_path):
    p = ROOT / rel_path
    if not p.exists():
        return f'<p style="color:#aaa;padding:20px;text-align:center;font-style:italic">Figure not found: {rel_path}</p>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;display:block;">'

def extract_section(desc, header):
    if not desc or not isinstance(desc, str):
        return ''
    tag = f'### {header}\n'
    idx = desc.find(tag)
    if idx == -1:
        return str(desc)[:280]
    start = idx + len(tag)
    end = desc.find('\n###', start)
    text = desc[start:end if end != -1 else None].strip()
    return text[:300] + ('…' if len(text) > 300 else '')

NAVY    = '#1B3A6B'
BLUE    = '#2E6DA4'
STEEL   = '#5B8BB9'
LIGHT   = '#ADC8E6'
ORANGE  = '#E07B2A'
RED     = '#B94040'
GRAY    = '#546E7A'
WHITE   = '#FFFFFF'
BG      = '#F2F5F9'

CPAL = [
    '#1B3A6B','#2E6DA4','#5B8BB9','#7BADD0',
    '#00695C','#00897B','#26A69A','#4DB6AC',
    '#4527A0','#7E57C2','#9575CD','#B39DDB',
    '#2E7D32','#558B2F','#8D6E63','#6D4C41',
    '#546E7A','#607D8B','#78909C','#90A4AE',
    '#BF5700','#B94040','#7B6B8D','#A0522D',
]

BASE = dict(plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            font=dict(family='Inter, Segoe UI, Arial, sans-serif',
                      size=12, color='#1C2B3A'))

def ax(**kw):
    return dict(gridcolor='#E8ECF0', linecolor='#D4DFE9',
                tickcolor='#D4DFE9', **kw)

def cfg():
    return {'responsive': True, 'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d','lasso2d','autoScale2d']}

def to_html(fig, fid=None):
    kw = dict(full_html=False, include_plotlyjs=False, config=cfg())
    if fid:
        kw['div_id'] = fid
    return pio.to_html(fig, **kw)

LANG = {
    'en':'English','es':'Spanish','fr':'French','pt':'Portuguese',
    'pt-PT':'Portuguese (PT)','de':'German','it':'Italian','ru':'Russian',
    'ar':'Arabic','tr':'Turkish','zh-CN':'Chinese (S)','zh-TW':'Chinese (T)',
    'vi':'Vietnamese','ja':'Japanese','ko':'Korean','hi':'Hindi',
    'hi-Latn':'Hindi (Latin)','fa':'Persian','pl':'Polish','nl':'Dutch',
    'cs':'Czech','ro':'Romanian','sv':'Swedish','da':'Danish','no':'Norwegian',
    'hu':'Hungarian','hr':'Croatian','bg':'Bulgarian','el':'Greek',
    'sk':'Slovak','uk':'Ukrainian','bs':'Bosnian','th':'Thai',
    'id':'Indonesian','ms':'Malay','bn':'Bengali','ur':'Urdu',
    'fil':'Filipino','tl':'Tagalog','az':'Azerbaijani','hy':'Armenian',
    'kk':'Kazakh','uz':'Uzbek','iw':'Hebrew','so':'Somali',
}

print('Loading from BigQuery…')

n_conversations = 46445
try:
    r = load(f"SELECT COUNT(*) AS n FROM `{bq_table('conversation_docs')}`")
    n_conversations = int(r['n'].iloc[0])
except Exception as e:
    print(f'  [WARN] conversation_docs: {e}')

weekly = load(f"""
    SELECT week_start AS week, total_conversations AS total,
           escalated_count, negative_ratings, positive_ratings,
           headline_failures, headline_failure_rate_pct, escalation_rate_pct,
           confirmed_failures, agent_requests, repetition_loops,
           english_conversations, non_english_conversations
    FROM `{bq_table('weekly_volume')}` ORDER BY week_start
""")
weekly['week'] = pd.to_datetime(weekly['week'])
weekly['roll4']      = weekly['total'].rolling(4, min_periods=1).mean()
weekly['roll4_fail'] = weekly['headline_failure_rate_pct'].rolling(4, min_periods=1).mean()
print(f'  Weekly: {len(weekly)} rows')

kpi_chunk = try_load(f"""
    SELECT COUNT(*) AS n_chunks,
           COUNT(DISTINCT IF(cluster_id>=0,cluster_id,NULL)) AS n_clusters,
           COUNTIF(is_noise) AS n_noise
    FROM `{bq_table('chunk_cluster_labels')}`
""", ['n_chunks','n_clusters','n_noise'])
n_chunks   = int(kpi_chunk['n_chunks'].iloc[0]   or 0) if not kpi_chunk.empty else 169709
n_clusters = int(kpi_chunk['n_clusters'].iloc[0] or 0) if not kpi_chunk.empty else 24
n_noise    = int(kpi_chunk['n_noise'].iloc[0]    or 0) if not kpi_chunk.empty else 20346
noise_pct  = round(100 * n_noise / max(n_chunks, 1), 1)

total_failures = int(weekly['headline_failures'].sum())
avg_fail_rate  = round(weekly['headline_failure_rate_pct'].mean(), 1)
avg_esc_rate   = round(weekly['escalation_rate_pct'].mean(), 1)

met = try_load(f"""
    SELECT sil_cos_all, dbcv, davies_bouldin, calinski_harabasz
    FROM `{bq_table('chunk_cluster_metrics')}` ORDER BY computed_at DESC LIMIT 1
""", ['sil_cos_all','dbcv','davies_bouldin','calinski_harabasz'])
sil_cos  = float(met['sil_cos_all'].iloc[0]      or 0.804) if not met.empty else 0.804
dbcv_val = float(met['dbcv'].iloc[0]             or 0.547) if not met.empty else 0.547
db_val   = float(met['davies_bouldin'].iloc[0]   or 0.240) if not met.empty else 0.240
ch_val   = float(met['calinski_harabasz'].iloc[0]or 382279) if not met.empty else 382279

fail_df = try_load(f"""
    SELECT cluster_id, n_chunks, total_conversations, headline_failures,
           chunk_headline_failure_rate_pct, chunk_escalation_rate_pct,
           explicit_negative_bot, repetition_loop, agent_request_unserved, abandoned_inquiry
    FROM `{bq_table('failure_by_chunk_cluster')}` ORDER BY cluster_id
""", ['cluster_id','n_chunks','chunk_headline_failure_rate_pct'])
print(f'  Failure rows: {len(fail_df)}')

label_df = try_load(f"""
    SELECT lt.cluster_id, cd.cluster_title AS title,
           lt.n_chunks, lt.n_conversations, lt.avg_hdbscan_prob,
           lt.chunk_headline_failure_rate_pct, lt.chunk_escalation_rate_pct,
           lt.avg_chunks_per_conv, lt.is_routing_cluster,
           cd.cluster_description AS description
    FROM `{bq_table('chunk_cluster_label_table')}` lt
    LEFT JOIN `{bq_table('chunk_cluster_descriptions')}` cd USING (cluster_id)
    ORDER BY lt.chunk_headline_failure_rate_pct DESC
""", ['cluster_id','title','n_chunks','chunk_headline_failure_rate_pct'])
print(f'  Label rows: {len(label_df)}')
if not label_df.empty:
    label_df['theme']  = label_df['description'].apply(lambda d: extract_section(d, 'Theme'))
    label_df['action'] = label_df['description'].apply(lambda d: extract_section(d, 'Suggested action'))

fail_lang = load_csv('failure_by_language_multilingual.csv')
fc_cmp    = load_csv('forecasting_model_comparison.csv')

title_map = {}
if not label_df.empty:
    for _, r in label_df.iterrows():
        title_map[int(r['cluster_id'])] = str(r.get('title') or '')

def clabel(cid, n=38):
    t = title_map.get(int(cid), '')
    return f"C{cid}: {t[:n]}{'…' if len(t)>n else ''}" if t else f'C{cid}'

fig_vol = make_subplots(specs=[[{'secondary_y': True}]])
fig_vol.add_trace(go.Bar(
    x=weekly['week'], y=weekly['total'], name='Weekly conversations',
    marker_color=LIGHT, opacity=0.85), secondary_y=False)
fig_vol.add_trace(go.Scatter(
    x=weekly['week'], y=weekly['roll4'], name='4-week avg (volume)',
    line=dict(color=NAVY, width=2.5)), secondary_y=False)
fig_vol.add_trace(go.Scatter(
    x=weekly['week'], y=weekly['roll4_fail'], name='Failure rate % (4-wk avg)',
    line=dict(color=RED, width=2.5)), secondary_y=True)
fig_vol.add_trace(go.Scatter(
    x=weekly['week'], y=weekly['headline_failure_rate_pct'],
    name='Failure rate %', mode='markers',
    marker=dict(size=5, color=RED, opacity=0.45)), secondary_y=True)
fig_vol.update_layout(
    title=dict(text='Weekly volume and headline failure rate',
               x=0.01, xanchor='left', font=dict(size=14)),
    legend=dict(orientation='h', y=-0.18, x=0, xanchor='left',
                yanchor='top', font=dict(size=11)),
    margin=dict(t=50, b=100, l=60, r=60), **BASE)
fig_vol.update_xaxes(**ax())
fig_vol.update_yaxes(title_text='Conversations', **ax(), secondary_y=False)
fig_vol.update_yaxes(title_text='Failure rate (%)', showgrid=False,
                     tickcolor='#D4DFE9', secondary_y=True)

fig_fail = go.Figure()
if not fail_df.empty:
    fs = fail_df.sort_values('chunk_headline_failure_rate_pct', ascending=True).copy()
    fs['ylabel'] = fs['cluster_id'].apply(lambda x: clabel(int(x), 40))
    mean_f = fs['chunk_headline_failure_rate_pct'].mean()
    top3   = set(fs.nlargest(3, 'chunk_headline_failure_rate_pct')['cluster_id'])
    cols   = [NAVY if int(r['cluster_id']) in top3 else STEEL
              for _, r in fs.iterrows()]
    fig_fail.add_trace(go.Bar(
        y=fs['ylabel'], x=fs['chunk_headline_failure_rate_pct'],
        orientation='h', marker_color=cols, opacity=0.88,
        text=fs['chunk_headline_failure_rate_pct'].apply(lambda v: f'{v:.1f}%'),
        textposition='outside', textfont=dict(size=10, color='#1C2B3A'),
        customdata=fs[['n_chunks','total_conversations',
                        'headline_failures','chunk_escalation_rate_pct']].values,
        hovertemplate='<b>%{y}</b><br>'
            'Failure rate: <b>%{x:.1f}%</b><br>'
            'Chunks: %{customdata[0]:,}<br>'
            'Conversations: %{customdata[1]:,}<br>'
            'Total failures: %{customdata[2]:,}<br>'
            'Escalation: %{customdata[3]:.1f}%<extra></extra>'))
    fig_fail.add_vline(x=mean_f, line_dash='dash', line_color=GRAY, line_width=1.5,
        annotation_text=f'Avg {mean_f:.1f}%', annotation_font_color=GRAY,
        annotation_position='top right')
fig_fail.update_layout(
    title='Headline failure rate by intent cluster',
    xaxis_title='Chunk-weighted failure rate (%)',
    height=680, margin=dict(l=330, t=60, b=50, r=90),
    **BASE)
fig_fail.update_xaxes(**ax())
fig_fail.update_yaxes(**ax(automargin=True))

fig_bubble = go.Figure()
if not label_df.empty:
    b = label_df.copy()
    b['lbl'] = b['cluster_id'].apply(lambda x: f'C{int(x)}')
    fig_bubble.add_trace(go.Scatter(
        x=b['n_chunks'], y=b['chunk_headline_failure_rate_pct'],
        mode='markers+text', text=b['lbl'],
        textposition='top center',
        textfont=dict(size=9, color=NAVY),
        marker=dict(
            size=b['n_conversations'] / 140,
            color=b['chunk_headline_failure_rate_pct'],
            colorscale=[[0, LIGHT],[0.5, BLUE],[1, NAVY]],
            showscale=True,
            colorbar=dict(title='Fail %', thickness=13,
                          tickfont=dict(size=10)),
            opacity=0.85, line=dict(width=1.5, color='white')),
        customdata=b[['n_conversations','chunk_escalation_rate_pct',
                       'title']].values,
        hovertemplate='<b>%{text} — %{customdata[2]}</b><br>'
            'Chunks: %{x:,}<br>'
            'Failure rate: <b>%{y:.1f}%</b><br>'
            'Conversations: %{customdata[0]:,}<br>'
            'Escalation: %{customdata[1]:.1f}%<extra></extra>'))
fig_bubble.update_layout(
    title='Risk matrix — size ∝ volume · color = failure rate',
    xaxis_title='Chunks in cluster', yaxis_title='Failure rate (%)',
    height=500, margin=dict(t=60, b=50, l=60, r=30),
    **BASE)
fig_bubble.update_xaxes(**ax())
fig_bubble.update_yaxes(**ax())

fig_lang = go.Figure()
if not fail_lang.empty and 'detected_language' in fail_lang.columns:
    fl = fail_lang.copy()
    if 'reliable' in fl.columns:
        reliable = fl[fl['reliable'] == True]
        fl = reliable if not reliable.empty else fl
    fl = fl.sort_values('fail_rate', ascending=True).copy()
    fl['lang_name'] = fl['detected_language'].apply(
        lambda c: LANG.get(c, c.replace('-', ' ').title()))
    avg_lr = fl['fail_rate'].mean()
    top3l  = set(fl.nlargest(3, 'fail_rate')['detected_language'])
    colors = [NAVY if c in top3l else STEEL
              for c in fl['detected_language']]
    cd_cols = [c for c in ['total','failures','esc_rate'] if c in fl.columns]
    fig_lang.add_trace(go.Bar(
        y=fl['lang_name'], x=fl['fail_rate'], orientation='h',
        marker_color=colors, opacity=0.88,
        text=fl['fail_rate'].apply(lambda v: f'{v:.1f}%'),
        textposition='outside', textfont=dict(size=10),
        customdata=fl[cd_cols].values if cd_cols else None,
        hovertemplate=(
            '<b>%{y}</b><br>Failure rate: <b>%{x:.1f}%</b>'
            + (f'<br>Conversations: %{{customdata[0]:,}}' if cd_cols else '')
            + '<extra></extra>')))
    fig_lang.add_vline(x=avg_lr, line_dash='dash', line_color=GRAY, line_width=1.5,
        annotation_text=f'Avg {avg_lr:.1f}%', annotation_font_color=GRAY,
        annotation_position='top right')
fig_lang.update_layout(
    title='Headline Failure Rate by Language',
    xaxis_title='Failure rate (%)',
    height=max(400, 28 * len(fl) + 80) if not fail_lang.empty else 400,
    margin=dict(l=160, t=60, b=50, r=90),
    **BASE)
fig_lang.update_xaxes(**ax())
fig_lang.update_yaxes(**ax(automargin=True))

fig_fc = go.Figure()
if not fc_cmp.empty and 'cluster_id' in fc_cmp.columns:
    fct = fc_cmp[['cluster_id','best_model','prophet_mae','ets_mae',
                   'sarima_mae','naive_last_mae']].copy()
    fct['cluster'] = fct['cluster_id'].apply(lambda x: clabel(int(x), 44))
    row_fills = []
    for bm in fct['best_model']:
        if bm == 'Prophet':     row_fills.append('#E8F0F9')
        elif bm == 'Ets':       row_fills.append('#EEF4F0')
        elif bm == 'Sarima':    row_fills.append('#F4F0EE')
        else:                   row_fills.append('white')
    fig_fc = go.Figure(data=[go.Table(
        header=dict(
            values=['Cluster','Best model','Prophet MAE',
                    'ETS MAE','SARIMA MAE','Naive-last MAE'],
            fill_color=NAVY,
            font=dict(color='white', size=11,
                      family='Inter, Segoe UI, Arial'),
            align='left', height=36),
        cells=dict(
            values=[fct['cluster'], fct['best_model'],
                    fct['prophet_mae'].round(1), fct['ets_mae'].round(1),
                    fct['sarima_mae'].round(1), fct['naive_last_mae'].round(1)],
            align='left',
            font=dict(size=11, family='Inter, Segoe UI, Arial'),
            fill_color=[row_fills]*6,
            height=30))])
    fig_fc.update_layout(
        title='Forecasting accuracy — weekly CV, 23 folds (MAE = chunks/day)',
        height=max(280, 50 + 32 * len(fct)),
        margin=dict(t=50, b=10, l=16, r=16),
        paper_bgcolor=WHITE,
        font=dict(family='Inter, Segoe UI, Arial'))

eval_rows = [
    ['Chunk HDBSCAN + fine-tuned MiniLM-L12-v2','24',
     f'{noise_pct:.0f}%', f'{sil_cos:.3f}',
     f'{db_val:.3f}', f'{ch_val:,.0f}', f'{dbcv_val:.3f}'],
    ['TF-IDF + K-Means (LSA-100D)',      '24','0%','0.143','2.624','3,714','—'],
    ['TF-IDF + Ward (LSA-100D)',          '24','0%','0.147','2.483','3,791','—'],
    ['Conv-level HDBSCAN (multilingual)', '70','34%','0.077','0.593','29,188','0.322'],
    ['Conv-level HDBSCAN (translated)',   '70','37%','0.091','0.559','28,878','0.462'],
]
fig_eval = go.Figure(data=[go.Table(
    header=dict(
        values=['Method','k','Noise','Silhouette ↑','Davies-Bouldin ↓',
                'Calinski-H ↑','DBCV ↑'],
        fill_color=NAVY,
        font=dict(color='white', size=11, family='Inter, Segoe UI, Arial'),
        align='left', height=36),
    cells=dict(
        values=[[r[i] for r in eval_rows] for i in range(7)],
        fill_color=[['#E8F0F9','white','white','white','white']]*7,
        align='left',
        font=dict(size=11, family='Inter, Segoe UI, Arial'),
        height=32))])
fig_eval.update_layout(
    title='Clustering evaluation',
    height=290, margin=dict(t=50, b=10, l=16, r=16),
    paper_bgcolor=WHITE,
    font=dict(family='Inter, Segoe UI, Arial'))

def cluster_table_html(df):
    if df.empty:
        return '<p style="color:#999;padding:20px">No data.</p>'
    rows = []
    for _, r in df.iterrows():
        cid    = int(r['cluster_id'])
        title  = str(r.get('title') or f'Cluster {cid}')
        chunks = int(r['n_chunks'])
        convs  = int(r.get('n_conversations', 0))
        fail   = float(r['chunk_headline_failure_rate_pct'])
        esc    = float(r['chunk_escalation_rate_pct'])
        prob   = float(r.get('avg_hdbscan_prob', 0))
        theme  = str(r.get('theme') or '')
        action = str(r.get('action') or '')
        routing= bool(r.get('is_routing_cluster', False))

        # Subtle left border colour only — no background fills
        if   fail >= 18: border = '#B94040'
        elif fail >= 14: border = STEEL
        else:            border = '#D4DFE9'

        badge = (f' <span style="font-size:0.68rem;background:#E8F0F9;'
                 f'color:{NAVY};padding:1px 7px;border-radius:10px;'
                 f'vertical-align:middle">routing</span>'
                 if routing else '')

        rows.append(f"""
  <tr class="cr" onclick="toggle(this)">
    <td style="font-weight:700;color:{NAVY};white-space:nowrap">C{cid}</td>
    <td style="font-weight:600">{title}{badge}</td>
    <td style="text-align:right">{chunks:,}</td>
    <td style="text-align:right">{convs:,}</td>
    <td style="text-align:right;font-weight:700;color:{NAVY}">{fail:.1f}%</td>
    <td style="text-align:right">{esc:.1f}%</td>
    <td style="text-align:right;color:#888">{prob:.3f}</td>
    <td style="text-align:center;color:#ccc">▼</td>
  </tr>
  <tr class="dr" style="display:none">
    <td colspan="8" style="padding:12px 18px 14px 50px;
        border-left:3px solid {border};background:#FAFBFD;
        font-size:0.85rem;line-height:1.65;color:#333">
      <div><span style="font-weight:600;color:{NAVY}">Intent:</span> {theme}</div>
      {'<div style="margin-top:5px"><span style="font-weight:600;color:'+GRAY+'">Action:</span> '+action+'</div>' if action else ''}
    </td>
  </tr>""")

    return f"""<div style="display:flex;gap:12px;align-items:center;margin-bottom:14px">
  <input id="cs" oninput="filterC()" placeholder="Search clusters…"
    style="padding:8px 14px;border:1px solid #D4DFE9;border-radius:6px;
           font-size:0.88rem;width:240px;font-family:inherit;color:#1C2B3A">
  <span style="color:#999;font-size:0.78rem">Click a row to expand Gemini analysis · click headers to sort</span>
</div>
<div style="overflow-x:auto;border-radius:8px;border:1px solid #D4DFE9">
<table id="ct" style="width:100%;border-collapse:collapse;font-size:0.86rem">
  <thead><tr style="background:{NAVY};color:white">
    <th style="padding:11px 14px;text-align:left;cursor:pointer;white-space:nowrap" onclick="sortC(0)">C#</th>
    <th style="padding:11px 14px;text-align:left;cursor:pointer" onclick="sortC(1)">Intent Cluster</th>
    <th style="padding:11px 14px;text-align:right;cursor:pointer" onclick="sortC(2)">Chunks</th>
    <th style="padding:11px 14px;text-align:right;cursor:pointer" onclick="sortC(3)">Convs</th>
    <th style="padding:11px 14px;text-align:right;cursor:pointer" onclick="sortC(4)">Fail %</th>
    <th style="padding:11px 14px;text-align:right;cursor:pointer" onclick="sortC(5)">Esc %</th>
    <th style="padding:11px 14px;text-align:right;cursor:pointer" onclick="sortC(6)">Conf.</th>
    <th style="padding:11px 14px;width:22px"></th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></div>"""

PLOTLYJS = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Automated Insights for Multilingual Support Chatbots</title>
{PLOTLYJS}
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter','Segoe UI',Arial,sans-serif;
     background:{BG};color:#1C2B3A;line-height:1.5;font-size:14px}}
a{{color:{BLUE}}}

header{{background:{NAVY};color:white;
        padding:24px 44px 18px;border-bottom:3px solid {ORANGE}}}
header h1{{font-size:1.4rem;font-weight:800;letter-spacing:-0.02em}}
header p{{margin-top:4px;font-size:0.80rem;opacity:0.68;line-height:1.6}}

.kpis{{display:flex;flex-wrap:wrap;gap:10px;
       padding:14px 44px;background:white;
       border-bottom:1px solid #D4DFE9}}
.kpi{{display:flex;flex-direction:column;align-items:center;
      justify-content:center;border-radius:8px;padding:10px 16px;
      min-width:100px;flex:1;max-width:140px;color:white}}
.kpi.n{{background:{NAVY}}}
.kpi.b{{background:{BLUE}}}
.kpi.s{{background:{STEEL}}}
.kpi.o{{background:{ORANGE}}}
.kpi .v{{font-size:1.6rem;font-weight:800;letter-spacing:-0.02em;line-height:1}}
.kpi .l{{font-size:0.60rem;opacity:0.80;margin-top:4px;letter-spacing:0.05em;
         text-align:center;text-transform:uppercase}}

.tabs{{display:flex;padding:0 44px;background:white;
       border-bottom:1px solid #D4DFE9;overflow-x:auto}}
.tab{{padding:12px 20px;border:none;background:transparent;cursor:pointer;
      font-size:0.87rem;font-weight:500;color:{GRAY};white-space:nowrap;
      border-bottom:2px solid transparent;margin-bottom:-1px;
      transition:color .15s;font-family:inherit}}
.tab:hover{{color:{NAVY}}}
.tab.on{{color:{NAVY};border-bottom-color:{NAVY};font-weight:700}}

.panel{{display:none;padding:28px 44px 56px}}
.panel.on{{display:block}}

.card{{background:white;border-radius:8px;border:1px solid #D4DFE9;
       padding:6px;margin-bottom:20px}}
.img-card{{background:white;border-radius:8px;border:1px solid #D4DFE9;
           padding:20px 22px;margin-bottom:20px;overflow:hidden}}
.img-card h4{{font-size:0.73rem;font-weight:700;color:{NAVY};margin-bottom:12px;
              text-transform:uppercase;letter-spacing:0.06em}}

.g2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}

.sec{{font-size:0.72rem;font-weight:700;color:{GRAY};text-transform:uppercase;
      letter-spacing:0.09em;margin:26px 0 12px;
      padding-bottom:6px;border-bottom:1px solid #D4DFE9}}
.sec:first-child{{margin-top:0}}

.prow{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}
.pbox{{background:{NAVY};color:white;border-radius:8px;padding:15px 18px;text-align:center}}
.pbox .pv{{font-size:1.45rem;font-weight:800}}
.pbox .pl{{font-size:0.65rem;opacity:0.75;margin-top:4px;text-transform:uppercase;letter-spacing:0.04em}}

#ct tbody tr.cr{{cursor:pointer;transition:background .1s}}
#ct tbody tr.cr:hover{{background:#F2F5F9}}
#ct tbody tr.cr td{{padding:10px 14px;border-bottom:1px solid #EEF1F6;white-space:nowrap}}
#ct tbody tr.dr td{{border-bottom:2px solid #D4DFE9}}

footer{{text-align:center;padding:22px 44px;color:#aaa;font-size:0.74rem;
        background:white;border-top:1px solid #D4DFE9;margin-top:20px}}

@media(max-width:860px){{
  .g2,.prow{{grid-template-columns:1fr}}
  .panel,.kpis,.tabs{{padding-left:16px;padding-right:16px}}
  header{{padding:18px 16px 14px}}
}}
</style>
</head>
<body>

<header>
  <h1>Automated Insights for Multilingual Support Chatbots</h1>
  <p>10Web AI Chatbot &nbsp;|&nbsp; May–November 2025 &nbsp;|&nbsp;
     46,445 conversations, 93 languages &nbsp;|&nbsp;
     Anna Minasyan, BSc Data Science, AUA &nbsp;|&nbsp;
     Supervisor: Armen Saghatelian, PhD / 10Web</p>
</header>

<div class="kpis">
  <div class="kpi n"><div class="v">{n_conversations:,}</div><div class="l">Conversations</div></div>
  <div class="kpi n"><div class="v">{n_chunks:,}</div><div class="l">Chunks</div></div>
  <div class="kpi n"><div class="v">{n_clusters}</div><div class="l">Intent clusters</div></div>
  <div class="kpi s"><div class="v">{noise_pct}%</div><div class="l">HDBSCAN noise</div></div>
  <div class="kpi o"><div class="v">{avg_fail_rate}%</div><div class="l">Avg failure rate</div></div>
  <div class="kpi o"><div class="v">{total_failures:,}</div><div class="l">Total failures</div></div>
  <div class="kpi b"><div class="v">{avg_esc_rate}%</div><div class="l">Avg escalation</div></div>
  <div class="kpi b"><div class="v">{sil_cos:.3f}</div><div class="l">Silhouette</div></div>
  <div class="kpi b"><div class="v">{dbcv_val:.3f}</div><div class="l">DBCV</div></div>
</div>

<nav class="tabs">
  <button class="tab on"  onclick="show('overview',this)">Overview</button>
  <button class="tab"     onclick="show('failures',this)">Failure Analysis</button>
  <button class="tab"     onclick="show('forecast',this)">Forecasting</button>
  <button class="tab"     onclick="show('method',this)">Methodology</button>
</nav>

<div id="overview" class="panel on">
  <p class="sec">Volume and Failure Rate Trend</p>
  <div class="card">{to_html(fig_vol,'vol')}</div>

  <p class="sec">Intent clusters</p>
  <div style="background:white;border-radius:8px;border:1px solid #D4DFE9;padding:18px 20px">
    {cluster_table_html(label_df)}
  </div>
</div>

<div id="failures" class="panel">
  <p class="sec">Failure Rate by Intent Cluster</p>
  <div class="card">{to_html(fig_fail,'fail')}</div>

  <p class="sec">Failure Signal Breakdown — Top 10 Clusters</p>
  <div class="img-card">
    <h4>unresolved · confirmed negative (T1) · agent escalation (T2)</h4>
    {embed_png('data/figures/failure_breakdown_top10.png')}
  </div>

  <p class="sec">Risk Matrix — Volume vs Failure Rate</p>
  <div class="card">{to_html(fig_bubble,'bubble')}</div>

  <p class="sec">Failure Rate by Language</p>
  <div class="card">{to_html(fig_lang,'lang')}</div>
</div>

<div id="forecast" class="panel">
  <p class="sec">Daily chunk forecast — C11 (Domain Purchase)</p>
  <div class="img-card">
    <h4>Prophet · weekly seasonality · 11-fold expanding CV · 95% interval</h4>
    {embed_png('data/figures/daily_chunk_forecast_plot.png')}
  </div>

  <p class="sec">Model Accuracy per Cluster</p>
  <div class="card">{to_html(fig_fc,'fc')}</div>
</div>

<div id="method" class="panel">
  <p class="sec">Pipeline Architecture</p>
  <div class="img-card">
    <h4>de-identification → chunking → fine-tuning → UMAP+HDBSCAN → failure analysis</h4>
    {embed_png('data/figures/pipeline_plot.png')}
  </div>

  <p class="sec">Key Results</p>
  <div class="prow">
    <div class="pbox"><div class="pv">169,709</div><div class="pl">Semantic chunks</div></div>
    <div class="pbox"><div class="pv">24</div><div class="pl">Intent clusters</div></div>
    <div class="pbox"><div class="pv">{sil_cos:.3f}</div><div class="pl">Silhouette (cosine 384-D)</div></div>
    <div class="pbox"><div class="pv">{dbcv_val:.3f}</div><div class="pl">DBCV (UMAP-10D)</div></div>
    <div class="pbox"><div class="pv">8.8×</div><div class="pl">vs conversation-level baseline</div></div>
    <div class="pbox"><div class="pv">12%</div><div class="pl">HDBSCAN noise</div></div>
    <div class="pbox"><div class="pv">0.736</div><div class="pl">Prophet MASE (daily)</div></div>
    <div class="pbox"><div class="pv">−29%</div><div class="pl">vs naive wMAE</div></div>
  </div>

  <p class="sec">Contrastive Fine-Tuning — Before and After</p>
  <div class="g2">
    <div class="img-card">
      <h4>Base MiniLM-L12-v2 — before fine-tuning (noise ≈ 47%, 23 clusters)</h4>
      {embed_png('data/figures/umap_scatter_base.png')}
    </div>
    <div class="img-card">
      <h4>After contrastive fine-tuning (noise = 12%, 24 clusters, Silhouette = 0.804)</h4>
      {embed_png('data/figures/umap_scatter_finetuned.png')}
    </div>
  </div>

  <p class="sec">Clustering Evaluation</p>
  <div class="card">{to_html(fig_eval,'eval')}</div>
</div>

<footer>
  Generated by scripts/build_dashboard.py &nbsp;·&nbsp;
  {n_chunks:,} chunks · {n_clusters} clusters ·
  Silhouette = {sil_cos:.4f} · DBCV = {dbcv_val:.4f}
  &nbsp;|&nbsp; American University of Armenia / 10Web · Capstone 2025–26
</footer>

<script>
function show(id, btn) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  if (btn) btn.classList.add('on');
  setTimeout(() => window.dispatchEvent(new Event('resize')), 80);
}}
function toggle(tr) {{
  const n = tr.nextElementSibling;
  if (!n || !n.classList.contains('dr')) return;
  const open = n.style.display !== 'none';
  n.style.display = open ? 'none' : 'table-row';
  const a = tr.querySelector('td:last-child');
  if (a) a.textContent = open ? '▼' : '▲';
}}
function filterC() {{
  const q = document.getElementById('cs').value.toLowerCase();
  document.querySelectorAll('#ct tbody tr.cr').forEach(tr => {{
    const m = tr.textContent.toLowerCase().includes(q);
    tr.style.display = m ? '' : 'none';
    const d = tr.nextElementSibling;
    if (d && d.classList.contains('dr')) d.style.display = 'none';
  }});
}}
let sd = {{}};
function sortC(c) {{
  const tb = document.querySelector('#ct tbody');
  const ps = [];
  let t = tb.firstElementChild;
  while (t) {{
    const d = t.nextElementSibling;
    if (t.classList.contains('cr'))
      ps.push({{cr:t, dr: d && d.classList.contains('dr') ? d : null}});
    t = t.nextElementSibling;
  }}
  const asc = !sd[c]; sd = {{}}; sd[c] = asc;
  ps.sort((a,b) => {{
    const va = a.cr.cells[c]?.textContent.trim()||'';
    const vb = b.cr.cells[c]?.textContent.trim()||'';
    const na = parseFloat(va.replace(/[^0-9.-]/g,''));
    const nb = parseFloat(vb.replace(/[^0-9.-]/g,''));
    if (!isNaN(na)&&!isNaN(nb)) return asc?na-nb:nb-na;
    return asc?va.localeCompare(vb):vb.localeCompare(va);
  }});
  ps.forEach(p => {{
    tb.appendChild(p.cr);
    if (p.dr) {{ p.dr.style.display='none'; tb.appendChild(p.dr); }}
  }});
}}
</script>
</body>
</html>"""

out = ROOT / 'dashboard.html'
out.write_text(html, encoding='utf-8')
size_mb = out.stat().st_size / 1e6
print(f'\nDashboard → {out}  ({size_mb:.1f} MB)')
print(f'Open: open "{out}"')
