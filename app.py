import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from pathlib import Path
import scipy.stats as stats
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="UFC Fighting Style Analysis",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide Streamlit default chrome */
  #MainMenu {visibility: hidden;}
  footer    {visibility: hidden;}

  /* Page background — dark charcoal */
  .stApp { background-color: #0f1117; }

  /* All default text to light */
  html, body, [class*="css"], .stMarkdown, .stMarkdown p,
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
  label, .stSelectbox label {
    color: #f1f5f9 !important;
  }

  /* Selectbox inputs */
  .stSelectbox > div > div {
    background-color: #1e2330 !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
  }

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #1e2330;
    padding: 10px 16px 0;
    border-radius: 14px 14px 0 0;
    border: 1px solid #334155;
    border-bottom: none;
  }

  /* Inactive tabs */
  .stTabs [data-baseweb="tab"] {
    background: #2d3748;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 14px;
    color: #94a3b8 !important;
    border: none;
  }

  /* Active tab */
  .stTabs [aria-selected="true"] {
    background: #e53e3e !important;
    color: #ffffff !important;
  }

  /* Tab panel content area */
  .stTabs [data-baseweb="tab-panel"] {
    background: #1e2330;
    border-radius: 0 0 14px 14px;
    border: 1px solid #334155;
    padding: 28px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
  }

  /* Note text below charts */
  .note-text {
    font-size: 13px;
    font-style: italic;
    color: #94a3b8;
    margin-top: 12px;
    padding: 10px 14px;
    background: #2d3748;
    border-radius: 8px;
    border-left: 3px solid #e53e3e;
  }

  /* Hero card */
  .hero-card {
    background: #1e2330;
    border-radius: 14px;
    border: 1px solid #334155;
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
    padding: 28px 32px;
    margin-bottom: 24px;
  }
  .hero-title {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 12px;
  }
  .hero-body {
    font-size: 15px;
    color: #cbd5e1;
    line-height: 1.6;
  }
  .tag {
    display: inline-block;
    background: #2d3748;
    color: #e2e8f0;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 13px;
    font-weight: 600;
    margin-right: 8px;
    border: 1px solid #4a5568;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def parse_x_of_y(series):
    landed, attempted = [], []
    for v in series.astype(str).fillna("").str.strip():
        m = re.match(r"^(\d+)\s+of\s+(\d+)$", v)
        if m:
            landed.append(float(m.group(1)))
            attempted.append(float(m.group(2)))
        else:
            landed.append(np.nan)
            attempted.append(np.nan)
    return pd.Series(landed, index=series.index), pd.Series(attempted, index=series.index)

def parse_percent(series):
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    ) / 100

def parse_control_time(series):
    out = []
    for v in series.astype(str).fillna("").str.strip():
        m = re.match(r"^(\d+):(\d+)$", v)
        out.append(int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan)
    return pd.Series(out, index=series.index)

def parse_round_time(round_series, time_series):
    out = []
    for rv, tv in zip(round_series, time_series):
        try:
            rn = int(rv)
        except Exception:
            out.append(np.nan); continue
        m = re.match(r"^(\d+):(\d+)$", str(tv).strip())
        if not m:
            out.append(np.nan); continue
        out.append(max(rn - 1, 0) * 300 + int(m.group(1)) * 60 + int(m.group(2)))
    return pd.Series(out, index=round_series.index)

def win_rate_by_bin(data, feature, bins, min_fights=5):
    temp = data[[feature, "RedWin"]].dropna().copy()
    temp["bin"] = pd.cut(temp[feature], bins=bins, include_lowest=True)
    summary = (
        temp.groupby("bin", observed=False)
        .agg(win_rate=("RedWin", "mean"), fights=("RedWin", "size"))
        .reset_index()
    )
    summary["midpoint"] = [(i.left + i.right) / 2 for i in summary["bin"]]
    return summary[summary["fights"] >= min_fights]


# ══════════════════════════════════════════════════════════════
# DATA LOADING — cached
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    raw_df = pd.read_csv("merged_stats_n_scorecards.csv", low_memory=False)
    df = raw_df.copy()

    df["RedWin"] = df["red_fighter_result"].astype(str).str.upper().map({"W": 1, "L": 0})
    df["event_date_parsed"] = pd.to_datetime(df["event_date"], dayfirst=True, errors="coerce")
    df["event_year"] = df["event_date_parsed"].dt.year

    stat_groups = {
        "sig": "fighter_sig_str", "total": "fighter_total_str",
        "td": "fighter_TD", "head": "fighter_sig_str_head",
        "body": "fighter_sig_str_body", "leg": "fighter_sig_str_leg",
        "distance": "fighter_sig_str_distance", "clinch": "fighter_sig_str_clinch",
        "ground": "fighter_sig_str_ground",
    }

    for side_prefix, side_name in [("red_", "red"), ("blue_", "blue")]:
        for short_name, stem in stat_groups.items():
            landed, attempted = parse_x_of_y(df[f"{side_name}_{stem}"])
            df[f"{side_prefix}{short_name}_landed"] = landed
            df[f"{side_prefix}{short_name}_att"]    = attempted
        df[f"{side_prefix}sig_pct"]  = parse_percent(df[f"{side_name}_fighter_sig_str_pct"])
        df[f"{side_prefix}td_pct"]   = parse_percent(df[f"{side_name}_fighter_TD_pct"])
        df[f"{side_prefix}ctrl_sec"] = parse_control_time(df[f"{side_name}_fighter_ctrl"])
        df[f"{side_prefix}kd"]       = pd.to_numeric(df[f"{side_name}_fighter_KD"], errors="coerce")
        df[f"{side_prefix}sub_att"]  = pd.to_numeric(df[f"{side_name}_fighter_sub_att"], errors="coerce")
        df[f"{side_prefix}rev"]      = pd.to_numeric(df[f"{side_name}_fighter_rev"], errors="coerce")

    for stat in [
        "sig_landed","sig_att","total_landed","total_att","td_landed","td_att",
        "head_landed","body_landed","leg_landed","distance_landed","clinch_landed",
        "ground_landed","ctrl_sec","kd","sub_att","rev",
    ]:
        df[f"{stat}_diff"] = df[f"red_{stat}"] - df[f"blue_{stat}"]

    df["sig_pct_diff"] = df["red_sig_pct"] - df["blue_sig_pct"]
    df["td_pct_diff"]  = df["red_td_pct"]  - df["blue_td_pct"]

    df["method_group"] = np.select(
        [
            df["method"].astype(str).str.contains("KO|TKO", case=False, na=False),
            df["method"].astype(str).str.contains("Submission", case=False, na=False),
            df["method"].astype(str).str.contains("Decision", case=False, na=False),
        ],
        ["KO/TKO", "Submission", "Decision"],
        default="Other",
    )

    df["weight_class"] = (
        df["bout_type"].astype(str)
        .str.extract(r"([A-Za-z\s]+weight)", expand=False)
        .fillna("Other")
    )
    df["title_bout"]            = df["bout_type"].astype(str).str.contains("Title", case=False, na=False).astype(int)
    df["fight_elapsed_seconds"] = parse_round_time(df["round"], df["time"])

    # Fighter summary for heatmap
    red_view = pd.DataFrame({
        "fighter": raw_df["red_fighter_name"],
        "win": (df["RedWin"] == 1).astype(float),
        "sig_landed": df["red_sig_landed"], "sig_attempted": df["red_sig_att"],
        "head_landed": df["red_head_landed"], "body_landed": df["red_body_landed"],
        "leg_landed": df["red_leg_landed"],
    })
    blue_view = pd.DataFrame({
        "fighter": raw_df["blue_fighter_name"],
        "win": (df["RedWin"] == 0).astype(float),
        "sig_landed": df["blue_sig_landed"], "sig_attempted": df["blue_sig_att"],
        "head_landed": df["blue_head_landed"], "body_landed": df["blue_body_landed"],
        "leg_landed": df["blue_leg_landed"],
    })
    fighter_df = pd.concat([red_view, blue_view], ignore_index=True)

    fs = (
        fighter_df.groupby("fighter", dropna=True)
        .agg(fights=("win","size"), wins=("win","sum"),
             sig_landed=("sig_landed","sum"), sig_attempted=("sig_attempted","sum"),
             head_landed=("head_landed","sum"), body_landed=("body_landed","sum"),
             leg_landed=("leg_landed","sum"))
        .reset_index()
    )
    fs["losses"]        = fs["fights"] - fs["wins"]
    fs["win_rate"]      = fs["wins"]   / fs["fights"]
    fs["sig_per_fight"] = fs["sig_landed"] / fs["fights"]
    fs["accuracy"]      = fs["sig_landed"] / fs["sig_attempted"]
    fs["location_total"]= fs[["head_landed","body_landed","leg_landed"]].sum(axis=1)
    for zone in ["head","body","leg"]:
        fs[f"{zone}_pct"] = fs[f"{zone}_landed"] / fs["location_total"]
    fs = fs[fs["fights"] >= 5].sort_values("fighter").reset_index(drop=True)

    oh = float(df["red_head_landed"].fillna(0).sum() + df["blue_head_landed"].fillna(0).sum())
    ob = float(df["red_body_landed"].fillna(0).sum() + df["blue_body_landed"].fillna(0).sum())
    ol = float(df["red_leg_landed"].fillna(0).sum()  + df["blue_leg_landed"].fillna(0).sum())
    ot = oh + ob + ol
    overall_stats = {"head_pct": oh/ot, "body_pct": ob/ot, "leg_pct": ol/ot,
                     "sig_per_fight": 44.2, "accuracy": 0.431, "win_rate": 0.650, "fights": 7756}

    return raw_df, df, fs, overall_stats

raw_df, df, fighter_summary, overall_stats = load_data()


# ══════════════════════════════════════════════════════════════
# PRE-LOAD IMAGE (used in Intro tab)
# ══════════════════════════════════════════════════════════════
img_path = Path("fight_picture.png")
img_b64  = None
if img_path.exists():
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

# ══════════════════════════════════════════════════════════════
# HERO — title + thesis only (image lives in Intro tab)
# ══════════════════════════════════════════════════════════════
# Load UFC logo (transparent version generated at startup)
_logo_b64 = ""
_logo_path = Path("UFC-logo-transparent.png")
if not _logo_path.exists():
    # Generate transparent logo from original if not already done
    try:
        from PIL import Image as _PIL
        import numpy as _np
        _img = _PIL.open("UFC-logo.png").convert("RGBA")
        _data = _np.array(_img)
        _black = (_data[:,:,0]<60) & (_data[:,:,1]<60) & (_data[:,:,2]<60)
        _data[_black, 3] = 0
        _result = _PIL.fromarray(_data)
        _result = _result.crop(_result.getbbox())
        _result.save(str(_logo_path))
    except Exception:
        pass

if _logo_path.exists():
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()

_logo_html = (
    f"<img src='data:image/png;base64,{_logo_b64}' "
    f"style='height:70px;object-fit:contain;flex-shrink:0;'/>"
    if _logo_b64 else ""
)

st.markdown(f"""
<div class='hero-card'>
  <div style='display:flex;justify-content:space-between;
              align-items:center;margin-bottom:14px;'>
    <div class='hero-title' style='margin-bottom:0;'>
      UFC Fighting Style Analysis
    </div>
    {_logo_html}
  </div>
  <div class='hero-body'>
    <strong>Thesis:</strong> In the UFC, offensive striking edges are a stronger
    predictor of victory than grappling metrics.<br><br>
    <span class='tag'>7,756 fights</span>
    <span class='tag'>1994 – 2024</span>
    <br><br>
    <span class ='tag'>By: Saif Ansari, Leonardo Robles-Lara, Carlos Guajardo </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🎬 Intro",
    "🥊 Strike Zones",
    "🏆 Winners vs Losers",
    "📈 Style Evolution",
    "📊 Strike Differential",
    "⏱ Control Time",
    "🎯 Feature Ranking",
    "🤖 Model Proof",
])

# ══════════════════════════════════════════════════════════════
# TAB 0 — Intro
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### What Actually Wins a UFC Fight?")

    if img_b64:
        st.markdown(f"""
        <div style="border-radius:12px;overflow:hidden;margin-bottom:24px;
                    box-shadow:0 8px 28px rgba(0,0,0,0.5);position:relative;">
          <img src="data:image/png;base64,{img_b64}"
               style="width:100%;display:block;object-fit:cover;" />
        </div>
        """, unsafe_allow_html=True)
    else:
        # Placeholder when image file is not present
        st.markdown("""
        <div style="border-radius:12px;background:#2d3748;border:2px dashed #4a5568;
                    height:380px;display:flex;align-items:center;justify-content:center;
                    margin-bottom:24px;">
          <div style="text-align:center;color:#94a3b8;">
            <div style="font-size:48px;margin-bottom:12px;">🥊</div>
            <div style="font-size:16px;font-weight:600;">
              Add fight_picture.png to your project folder
            </div>
            <div style="font-size:13px;margin-top:8px;">
              Place the image file next to app.py and requirements.txt
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════
# TAB 1 — Body heatmap
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Visual 1. Body heatmap of significant strikes")

    def zone_color(pct):
        alpha = 0.20 + 0.85 * max(0, min(1, float(pct or 0)))
        return f"rgba(220,38,38,{alpha:.3f})"

    def build_silhouette(row, label):
        hp = float(row.get("head_pct", 0) or 0)
        bp = float(row.get("body_pct", 0) or 0)
        lp = float(row.get("leg_pct",  0) or 0)
        spf= float(row.get("sig_per_fight", 0) or 0)
        acc= float(row.get("accuracy", 0) or 0)
        wr = float(row.get("win_rate", 0) or 0)
        fi = row.get("fights", "")

        fig = go.Figure()
        fig.update_layout(
            shapes=[
                dict(type="circle",
                     x0=4.2,x1=5.8,y0=13.5,y1=17.0,
                     line=dict(color="#f1f5f9",width=2),
                     fillcolor=zone_color(hp)),
                dict(type="rect",
                     x0=3.7,x1=6.3,y0=7.5,y1=13.5,
                     line=dict(color="#f1f5f9",width=2),
                     fillcolor=zone_color(bp)),
                dict(type="rect",
                     x0=3.7,x1=5.0,y0=2.0,y1=7.5,
                     line=dict(color="#f1f5f9",width=2),
                     fillcolor=zone_color(lp)),
                dict(type="rect",
                     x0=5.05,x1=6.3,y0=2.0,y1=7.5,
                     line=dict(color="#f1f5f9",width=2),
                     fillcolor=zone_color(lp)),
                dict(type="line",x0=3.7,x1=2.3,y0=12.5,y1=9.0,
                     line=dict(color="#f1f5f9",width=2)),
                dict(type="line",x0=6.3,x1=7.7,y0=12.5,y1=9.0,
                     line=dict(color="#f1f5f9",width=2)),
            ],
            annotations=[
                dict(x=5.0,y=15.2,
                     text=f"Head<br>{hp*100:.1f}%",
                     showarrow=False,font=dict(size=18,color="white")),
                dict(x=5.0,y=10.3,
                     text=f"Body<br>{bp*100:.1f}%",
                     showarrow=False,font=dict(size=18,color="white")),
                dict(x=4.35,y=4.7,
                     text=f"Legs<br>{lp*100:.1f}%",
                     showarrow=False,font=dict(size=18,color="white")),
                dict(x=8.9,y=14.7,xanchor="left",align="left",
                     showarrow=False,
                     text=(f"<b>{label}</b><br>"
                           f"Sig strikes/fight: {spf:.1f}<br>"
                           f"Accuracy: {acc*100:.1f}%<br>"
                           f"Win rate: {wr*100:.1f}%"
                           + (f"<br>Fights: {int(fi)}" if fi else "")),
                     font=dict(size=15),
                     bgcolor="rgba(30,35,48,0.95)",
                     bordercolor="#475569",borderwidth=1,borderpad=10),
            ],
            xaxis=dict(visible=False, range=[1,13]),
            yaxis=dict(visible=False, range=[0,19]),
            height=520,
            margin=dict(l=20,r=220,t=20,b=20),
            paper_bgcolor="#1e2330",
            plot_bgcolor="#1e2330",
        )
        return fig

    fighter_names = ["Overall dataset"] + sorted(fighter_summary["fighter"].tolist())
    selected = st.selectbox("Search fighter (5+ fights):", fighter_names, index=0)

    if selected == "Overall dataset":
        row   = overall_stats
        label = "Overall dataset"
    else:
        row   = fighter_summary[fighter_summary["fighter"] == selected].iloc[0].to_dict()
        label = selected

    st.plotly_chart(build_silhouette(row, label), use_container_width=True)
    st.markdown("""<div class='note-text'>63% of all significant strikes land to the head —
    exactly where winners build their largest edge. Search a fighter to see how their
    targeting shifts in wins vs losses.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Winners vs Losers
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Visual 2. How winners build the striking edge")

    def get_wl(a, b):
        w = np.where(df["RedWin"]==1, a, b)
        l = np.where(df["RedWin"]==1, b, a)
        return w, l

    w_head,w_head_l   = get_wl(df["red_head_landed"],     df["blue_head_landed"])
    w_body,w_body_l   = get_wl(df["red_body_landed"],     df["blue_body_landed"])
    w_leg, w_leg_l    = get_wl(df["red_leg_landed"],      df["blue_leg_landed"])
    w_dist,w_dist_l   = get_wl(df["red_distance_landed"], df["blue_distance_landed"])
    w_clinch,w_cl_l   = get_wl(df["red_clinch_landed"],   df["blue_clinch_landed"])
    w_ground,w_gr_l   = get_wl(df["red_ground_landed"],   df["blue_ground_landed"])

    profiles = {
        "Target profile (Head / Body / Leg)": {
            "categories":    ["Head","Body","Leg"],
            "winner_values": [np.nanmean(w_head),np.nanmean(w_body),np.nanmean(w_leg)],
            "loser_values":  [np.nanmean(w_head_l),np.nanmean(w_body_l),np.nanmean(w_leg_l)],
        },
        "Range profile (Distance / Clinch / Ground*)": {
            "categories":    ["Distance","Clinch","Ground*"],
            "winner_values": [np.nanmean(w_dist),np.nanmean(w_clinch),np.nanmean(w_ground)],
            "loser_values":  [np.nanmean(w_dist_l),np.nanmean(w_cl_l),np.nanmean(w_gr_l)],
        },
    }

    profile_sel = st.selectbox("Profile:", list(profiles.keys()))
    p = profiles[profile_sel]
    w_vals = [round(v,1) for v in p["winner_values"]]
    l_vals = [round(v,1) for v in p["loser_values"]]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=p["categories"],y=w_vals,name="Winners",
                          marker_color="#1f77b4",text=w_vals,textposition="outside"))
    fig3.add_trace(go.Bar(x=p["categories"],y=l_vals,name="Losers",
                          marker_color="#d62728",text=l_vals,textposition="outside"))
    fig3.update_layout(template="plotly_dark",barmode="group",height=460,
                       margin=dict(l=50,r=30,t=20,b=80),
                       yaxis_title="Average significant strikes landed per fight",
                       yaxis=dict(range=[0,40],tickfont=dict(size=14),title_font=dict(size=15)),
                       xaxis=dict(tickfont=dict(size=15)),
                       legend=dict(font=dict(size=15)),
                       hoverlabel=dict(font_size=16,bgcolor="#1e2330",
                                       bordercolor="#475569",font_color="#f1f5f9"))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""<div class='note-text'>Winners land 11.7 more head strikes on average than
    losers — more than 10× the leg strike gap. It is not just volume, it is where.
    (* Ground strikes occur from a grappling position — see Visual 6.)</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Style Evolution
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Visual 3. Fighting style evolution through time")

    STYLE_ORDER  = ["Striking","Submission Grappling","Wrestling/Control","Balanced/Mixed","Low Activity/Unclear"]
    STYLE_COLORS = {"Striking":"#1f77b4","Submission Grappling":"#2ca02c",
                    "Wrestling/Control":"#72b7b2","Balanced/Mixed":"#f2cf5b",
                    "Low Activity/Unclear":"#b7b7b7"}

    def build_style_rows(side):
        return pd.DataFrame({
            "event_year": df["event_year"],
            "result":     df[f"{side}_fighter_result"].astype(str).str.upper(),
            "method":     df["method"].astype(str).str.lower(),
            "sig_distance_landed": df[f"{side}_distance_landed"],
            "sig_clinch_landed":   df[f"{side}_clinch_landed"],
            "sig_ground_landed":   df[f"{side}_ground_landed"],
            "kd":         df[f"{side}_kd"],
            "td_landed":  df[f"{side}_td_landed"],
            "td_attempted": df[f"{side}_td_att"],
            "sub_att":    df[f"{side}_sub_att"],
            "rev":        df[f"{side}_rev"],
            "ctrl_sec":   df[f"{side}_ctrl_sec"],
        })

    @st.cache_data
    def compute_style_share():
        ff = pd.concat([build_style_rows("red"),build_style_rows("blue")],ignore_index=True)
        ff["won"] = (ff["result"]=="W").astype(int)

        def safe(rows,col):
            return pd.to_numeric(rows[col],errors="coerce").fillna(0)

        rows = ff.copy()
        method = rows["method"].astype(str).str.lower()
        winner = rows["won"].eq(1)
        rows["striking_score"] = (
            safe(rows,"sig_distance_landed") + 0.8*safe(rows,"sig_clinch_landed")
            + 0.6*safe(rows,"sig_ground_landed") + 12*safe(rows,"kd"))
        rows["grappling_score"] = (
            6*safe(rows,"td_landed") + safe(rows,"td_attempted")
            + 8*safe(rows,"sub_att") + 4*safe(rows,"rev")
            + safe(rows,"ctrl_sec")/20 + 0.6*safe(rows,"sig_ground_landed"))
        rows.loc[winner & method.str.contains("submission",na=False),"grappling_score"] += 35
        rows.loc[winner & method.str.contains("ko|tko|doctor",regex=True,na=False),"striking_score"] += 35

        low  = rows[["striking_score","grappling_score"]].max(axis=1) < 6
        stk  = rows["striking_score"] >= rows["grappling_score"]*1.2
        grp  = rows["grappling_score"] >= rows["striking_score"]*1.2
        sub  = rows["sub_att"].gt(0)|(winner & method.str.contains("submission",na=False))
        rows["style"] = "Balanced/Mixed"
        rows.loc[stk,  "style"] = "Striking"
        rows.loc[grp & sub,  "style"] = "Submission Grappling"
        rows.loc[grp & ~sub, "style"] = "Wrestling/Control"
        rows.loc[low,  "style"] = "Low Activity/Unclear"

        yt = rows.groupby("event_year")["won"].size().reset_index(name="total").rename(columns={"event_year":"year"})
        ys = rows.groupby(["event_year","style"]).size().reset_index(name="count").rename(columns={"event_year":"year"})
        sh = ys.merge(yt,on="year")
        sh["share"] = (sh["count"]/sh["total"]*100).round(1)
        sh = sh[sh["total"]>=100].copy()
        return sh

    share_df   = compute_style_share()
    years_all  = sorted(share_df["year"].unique().tolist())

    fig7 = go.Figure()
    for style in STYLE_ORDER:
        sd      = share_df[share_df["style"]==style].set_index("year")
        y_vals  = [round(float(sd.loc[y,"share"]),1) if y in sd.index else 0.0 for y in years_all]
        n_vals  = [int(sd.loc[y,"count"]) if y in sd.index else 0 for y in years_all]
        t_vals  = [int(sd.loc[y,"total"]) if y in sd.index else 0 for y in years_all]
        fig7.add_trace(go.Scatter(
            x=years_all, y=y_vals, name=style,
            mode="lines", stackgroup="one",
            fillcolor=STYLE_COLORS[style],
            line=dict(color=STYLE_COLORS[style],width=0.5),
            customdata=list(zip(n_vals,t_vals)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Year: %{x}<br>"
                "Share: %{y:.1f}%<br>Appearances: %{customdata[0]} of %{customdata[1]}"
                "<extra></extra>"),
        ))
    fig7.update_layout(
        template="plotly_dark",height=500,
        margin=dict(l=60,r=20,t=20,b=100),
        xaxis=dict(title="Event year",dtick=2,tickfont=dict(size=14),title_font=dict(size=15)),
        yaxis=dict(title="Share of fighter appearances (%)",range=[0,100],ticksuffix="%",
                   tickfont=dict(size=14),title_font=dict(size=15)),
        legend=dict(orientation="h",y=-0.22,x=0,font=dict(size=14)),
        hoverlabel=dict(font_size=16,bgcolor="#1e2330",
                        bordercolor="#475569",font_color="#f1f5f9"),
        hovermode="x unified",
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("""<div class='note-text'>Striking grew from ~37% of fighters in the mid-2000s
    to over 65% today. The sport evolved toward the same dimension that predicts outcomes.
    Striking did not just predict victories — it became the sport.</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Strike Differential
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Visual 4. Significant strike differential and win rate")

    top_wc   = df["weight_class"].value_counts().head(8).index.tolist()
    wc_sel   = st.selectbox("Weight class:", ["All"] + top_wc, key="wc4")
    sig_bins = np.arange(-80, 85, 5)
    subset   = df if wc_sel=="All" else df[df["weight_class"]==wc_sel]
    summary  = win_rate_by_bin(subset,"sig_landed_diff",sig_bins,min_fights=5)
    summary["blue_win_rate"] = 1 - summary["win_rate"]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=summary["midpoint"],y=summary["win_rate"],
        mode="lines+markers",name="Red corner",
        line=dict(color="#d62728",width=2),marker=dict(color="#d62728",size=6),
        customdata=summary["fights"],
        hovertemplate="<b>Red corner</b><br>Differential: %{x}<br>Win rate: %{y:.1%}<br>Fights: %{customdata}<extra></extra>"))
    fig1.add_trace(go.Scatter(
        x=summary["midpoint"],y=summary["blue_win_rate"],
        mode="lines+markers",name="Blue corner",
        line=dict(color="#1f77b4",width=2,dash="dot"),marker=dict(color="#1f77b4",size=6),
        customdata=summary["fights"],
        hovertemplate="<b>Blue corner</b><br>Differential: %{x}<br>Win rate: %{y:.1%}<br>Fights: %{customdata}<extra></extra>"))
    fig1.add_hline(y=0.65,line_dash="dash",line_color="rgba(214,39,40,0.4)",line_width=1.5,
                   annotation_text="Red baseline 65%",annotation_position="top right",
                   annotation_font_size=14,annotation_font_color="rgba(214,39,40,0.9)")
    fig1.add_hline(y=0.35,line_dash="dash",line_color="rgba(31,119,180,0.4)",line_width=1.5,
                   annotation_text="Blue baseline 35%",annotation_position="bottom right",
                   annotation_font_size=14,annotation_font_color="rgba(31,119,180,0.9)")
    fig1.update_layout(template="plotly_dark",height=500,
                       margin=dict(l=50,r=30,t=20,b=50),
                       xaxis_title="Significant strike differential (Red − Blue)",
                       yaxis_title="Win rate",
                       yaxis=dict(range=[0,1],tickformat=".0%"),
                       hoverlabel=dict(font_size=16,bgcolor="#1e2330",
                                       bordercolor="#475569",font_color="#f1f5f9"),
                       xaxis=dict(tickfont=dict(size=14),title_font=dict(size=15)),
                       yaxis_title_font=dict(size=15),
                       legend=dict(font=dict(size=14)))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""<div class='note-text'>More significant strikes landed = higher chance of winning.
    The dashed line marks the 65% baseline from UFC ranking convention —
    everything above it is the striking edge at work.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — Control Time
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Visual 5. Control-time differential and win rate")

    method_sel    = st.selectbox("Finish method:", ["All","Decision","KO/TKO","Submission"], key="m5")
    control_bins  = np.arange(-720, 780, 60)
    subset2       = df if method_sel=="All" else df[df["method_group"]==method_sel]
    summary2      = win_rate_by_bin(subset2,"ctrl_sec_diff",control_bins,min_fights=5)
    summary2["blue_win_rate"] = 1 - summary2["win_rate"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=summary2["midpoint"],y=summary2["win_rate"],
        mode="lines+markers",name="Red corner",
        line=dict(color="#d62728",width=2),marker=dict(color="#d62728",size=6),
        customdata=summary2["fights"],
        hovertemplate="<b>Red corner</b><br>Ctrl diff: %{x}s<br>Win rate: %{y:.1%}<br>Fights: %{customdata}<extra></extra>"))
    fig2.add_trace(go.Scatter(
        x=summary2["midpoint"],y=summary2["blue_win_rate"],
        mode="lines+markers",name="Blue corner",
        line=dict(color="#1f77b4",width=2,dash="dot"),marker=dict(color="#1f77b4",size=6),
        customdata=summary2["fights"],
        hovertemplate="<b>Blue corner</b><br>Ctrl diff: %{x}s<br>Win rate: %{y:.1%}<br>Fights: %{customdata}<extra></extra>"))
    fig2.add_hline(y=0.65,line_dash="dash",line_color="rgba(214,39,40,0.4)",line_width=1.5,
                   annotation_text="Red baseline 65%",annotation_position="top right",
                   annotation_font_size=14,annotation_font_color="rgba(214,39,40,0.9)")
    fig2.add_hline(y=0.35,line_dash="dash",line_color="rgba(31,119,180,0.4)",line_width=1.5,
                   annotation_text="Blue baseline 35%",annotation_position="bottom right",
                   annotation_font_size=14,annotation_font_color="rgba(31,119,180,0.9)")
    fig2.update_layout(template="plotly_dark",height=500,
                       margin=dict(l=50,r=30,t=20,b=50),
                       xaxis_title="Control-time differential in seconds (Red − Blue)",
                       yaxis_title="Win rate",
                       yaxis=dict(range=[0,1],tickformat=".0%"),
                       hoverlabel=dict(font_size=16,bgcolor="#1e2330",
                                       bordercolor="#475569",font_color="#f1f5f9"),
                       xaxis=dict(tickfont=dict(size=14),title_font=dict(size=15)),
                       yaxis_title_font=dict(size=15),
                       legend=dict(font=dict(size=14)))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""<div class='note-text'>Control time predicts winning too — but more weakly
    and only in certain fight types. In KO/TKO fights the relationship nearly
    disappears entirely.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — Feature Ranking
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### Visual 6. Ranked feature strength (Spearman correlation)")

    feature_labels = {
        "sig_landed_diff":      ("Sig strikes landed diff",      "Striking"),
        "head_landed_diff":     ("Head strikes landed diff",     "Striking"),
        "body_landed_diff":     ("Body strikes landed diff",     "Striking"),
        "distance_landed_diff": ("Distance strikes landed diff", "Striking"),
        "clinch_landed_diff":   ("Clinch strikes landed diff",   "Striking"),
        "total_landed_diff":    ("Total strikes landed diff",    "Striking"),
        "sig_pct_diff":         ("Sig strike accuracy diff",     "Striking"),
        "kd_diff":              ("Knockdown diff",               "Striking"),
        "ground_landed_diff":   ("Ground strikes landed diff",   "Position-based"),
        "ctrl_sec_diff":        ("Control time diff",            "Grappling"),
        "td_landed_diff":       ("Takedowns landed diff",        "Grappling"),
        "sub_att_diff":         ("Submission attempts diff",     "Grappling"),
        "rev_diff":             ("Reversals diff",               "Grappling"),
    }
    family_colors = {"Striking":"#1f77b4","Position-based":"#ff7f0e","Grappling":"#2ca02c"}

    corr_rows = []
    for feat,(label,family) in feature_labels.items():
        valid = df[[feat,"RedWin"]].dropna()
        rho,pval = stats.spearmanr(valid[feat],valid["RedWin"])
        corr_rows.append({"label":label,"family":family,"abs_rho":abs(rho),"rho":rho,"pval":pval,"n":len(valid)})
    corr_df = pd.DataFrame(corr_rows).sort_values("abs_rho").reset_index(drop=True)

    family_sel = st.selectbox("Filter by family:", ["All","Striking","Position-based","Grappling"], key="f6")
    subset_corr = corr_df if family_sel=="All" else corr_df[corr_df["family"]==family_sel]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=subset_corr["abs_rho"], y=subset_corr["label"],
        orientation="h",
        marker_color=[family_colors[f] for f in subset_corr["family"]],
        customdata=list(zip(
            subset_corr["rho"].round(3),
            subset_corr["pval"].apply(lambda p: "<0.001" if p<0.001 else f"{p:.3f}"),
            subset_corr["family"],
        )),
        hovertemplate="<b>%{y}</b><br>ρ: %{customdata[0]}<br>p: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    fig4.update_layout(
        template="plotly_dark",height=540,
        margin=dict(l=220,r=30,t=20,b=60),
        xaxis_title="Absolute Spearman correlation with winning (ρ)",
        xaxis=dict(range=[0,0.7],tickfont=dict(size=14),title_font=dict(size=15)),
        yaxis=dict(tickfont=dict(size=14)),
        hoverlabel=dict(font_size=16,bgcolor="#1e2330",
                        bordercolor="#475569",font_color="#f1f5f9"),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""<div class='note-text'>Striking metrics dominate the top of the ranking.
    The strongest pure grappling metric — control time — sits below six striking features.
    All correlations p &lt; 0.001. Blue = Striking · Orange = Position-based · Green = Grappling
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 7 — Model Proof
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### Visual 7. Model-based test of the thesis")

    @st.cache_data
    def run_models():
        striking_features = [
            "sig_landed_diff","total_landed_diff","sig_pct_diff","kd_diff",
            "head_landed_diff","body_landed_diff","distance_landed_diff","clinch_landed_diff",
        ]
        grappling_features = ["ctrl_sec_diff","td_landed_diff","sub_att_diff","rev_diff"]
        context_features   = ["fight_elapsed_seconds","title_bout"]

        feature_sets = {
            "Striking only":  striking_features,
            "Grappling only": grappling_features,
            "Combined":       striking_features + grappling_features + context_features,
        }
        results = {}
        for name,feats in feature_sets.items():
            X = df[feats].copy(); y = df["RedWin"]
            valid = y.notna(); X = X.loc[valid]; y = y.loc[valid].astype(int)
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
            pipe = Pipeline([("imp",SimpleImputer(strategy="median")),("scl",StandardScaler())])
            X_tr = pipe.fit_transform(X_train); X_te = pipe.transform(X_test)
            m = LogisticRegression(max_iter=2000); m.fit(X_tr,y_train)
            probs = m.predict_proba(X_te)[:,1]; preds = (probs>=0.5).astype(int)
            cm = confusion_matrix(y_test,preds)
            tn,fp,fn,tp = cm.ravel()
            results[name] = {
                "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
                "total":int(tn+fp+fn+tp),"correct":int(tn+tp),"wrong":int(fp+fn),
                "accuracy":round(accuracy_score(y_test,preds),3),
                "f1":round(f1_score(y_test,preds),3),
                "roc_auc":round(roc_auc_score(y_test,probs),3),
            }
        return results

    cm_results = run_models()
    fs_names   = list(cm_results.keys())
    fs_sel     = st.selectbox("Feature set:", fs_names, key="fs7")
    r          = cm_results[fs_sel]

    st.markdown(
        f"**{fs_sel} — Logistic Regression** &nbsp;|&nbsp; "
        f"Correct: **{r['correct']:,} / {r['total']:,} ({r['accuracy']:.1%})** &nbsp;|&nbsp; "
        f"Naive baseline: 65.0% &nbsp;|&nbsp; "
        f"F1: {r['f1']} &nbsp;|&nbsp; ROC-AUC: {r['roc_auc']}"
    )

    z = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
    text = [
        [f"<b>{r['tn']:,}</b><br>({r['tn']/r['total']:.1%})<br>Correct",
         f"<b>{r['fp']:,}</b><br>({r['fp']/r['total']:.1%})<br>Wrong"],
        [f"<b>{r['fn']:,}</b><br>({r['fn']/r['total']:.1%})<br>Wrong",
         f"<b>{r['tp']:,}</b><br>({r['tp']/r['total']:.1%})<br>Correct"],
    ]
    fig5 = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted: Blue wins","Predicted: Red wins"],
        y=["Actual: Blue wins","Actual: Red wins"],
        text=text, texttemplate="%{text}",
        textfont=dict(size=22,color="white"),
        colorscale=[[0.0,"#d62728"],[0.5,"#d4a017"],[1.0,"#2ca02c"]],
        showscale=False,
        hovertemplate="<b>%{y} → %{x}</b><br>Count: %{z}<extra></extra>",
    ))
    fig5.update_layout(
        template="plotly_dark",height=480,
        margin=dict(l=150,r=30,t=20,b=80),
        xaxis=dict(title="Predicted outcome",side="bottom"),
        yaxis=dict(title="Actual outcome",autorange="reversed"),
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""<div class='note-text'>Green = correct prediction · Red = wrong.
    Striking alone correctly called 1,642 of 1,939 test fights (84.7%).
    Grappling alone managed only 1,437 (74.1%). Combined got 1,715 right (88.4%).
    Use the dropdown to switch between feature sets.</div>""", unsafe_allow_html=True)