"""
Visual design system for the workspace: a restrained, table/chart-first aesthetic (research software, not a slide
deck). ONE centralized theme: every colour comes from the TOKENS below (light or dark); pages never hard-code colours.
Nothing here touches backend logic.

How theming works
  * `init_theme()` decides the active theme (widget choice > session > URL ?theme= > default light).
  * `inject_css()` writes the tokens as CSS variables and the rules that use them (plus the dark overrides of
    Streamlit's native widgets when the theme is dark).
  * `styled_figure()` / `palette()` give every Plotly chart the same tokens, so no chart keeps a white background
    in dark mode and no page needs its own dark version.
  * `theme_toggle()` is the Light / Dark switch shown in the application header; `persist_theme()` remembers the
    choice across browser refreshes (localStorage), page navigation is covered by the session.
"""
import html as _html

import plotly.graph_objects as go
import streamlit as st

THEMES = ("light", "dark")
LABELS = {"light": "☀ Light", "dark": "🌙 Dark"}

# Light values equal the original design (.streamlit/config.toml). Dark is a genuine second set of tokens, not an inversion.
TOKENS = {
    "light": {
        "bg": "#fcfcfb", "surface": "#f2f1ec", "elevated": "#ffffff", "text": "#0b0b0b", "text2": "#55554f",
        "border": "rgba(120,120,115,0.30)", "input": "#ffffff", "chart_bg": "rgba(0,0,0,0)", "accent": "#2a78d6",
        "accent_text": "#ffffff", "info": "#1f66bd", "success": "#0a7a2f", "warning": "#8a5a00", "critical": "#b42318",
        "grid": "rgba(120,120,115,0.18)", "axis": "rgba(120,120,115,0.45)", "hover_bg": "#fcfcfb",
        "muted": "#8a8a86", "subtle": "#6b6b68", "highlight": "#c98500", "heat_low": "#f2f1ec", "shadow": "rgba(0,0,0,0.12)",
        "categorical": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    },
    "dark": {
        "bg": "#0B1120", "surface": "#111827", "elevated": "#1F2937", "text": "#F9FAFB", "text2": "#9CA3AF",
        "border": "rgba(156,163,175,0.30)", "input": "#0F172A", "chart_bg": "rgba(0,0,0,0)", "accent": "#5aa2f0",
        "accent_text": "#0B1120", "info": "#7cb7ff", "success": "#34d399", "warning": "#fbbf24", "critical": "#f87171",
        "grid": "rgba(156,163,175,0.18)", "axis": "rgba(156,163,175,0.45)", "hover_bg": "#1F2937",
        "muted": "#9CA3AF", "subtle": "#cbd5e1", "highlight": "#fbbf24", "heat_low": "#1F2937", "shadow": "rgba(0,0,0,0.55)",
        "categorical": ["#5aa2f0", "#fb923c", "#34d399", "#fbbf24", "#f9a8d4", "#4ade80", "#a78bfa", "#f87171"],
    },
}

# kept for backward compatibility with older imports (light values)
CATEGORICAL = TOKENS["light"]["categorical"]
STATUS = {"good": "#0ca30c", "warning": "#c98500", "serious": "#c9663f", "critical": "#d03b3b", "neutral": "#6b6b68"}
_DARK_STATUS = {"good": "#34d399", "warning": "#fbbf24", "serious": "#fb923c", "critical": "#f87171", "neutral": "#9CA3AF"}


# ------------------------------------------------------------------------------------------------ theme state
def _normalize(x) -> str | None:
    if x in THEMES:
        return x
    for k, v in LABELS.items():
        if x == v:
            return k
    return None


def init_theme() -> str:
    """Resolve the active theme once per run. Priority: the header toggle > session > ?theme= in the URL > light."""
    S = st.session_state
    picked = _normalize(S.get("theme_pick"))
    if picked:
        if picked != S.get("theme"):
            S["theme_explicit"] = True
        S["theme"] = picked
    elif "theme" not in S:
        from_url = _normalize(st.query_params.get("theme"))
        S["theme"] = from_url or "light"
        S["theme_explicit"] = from_url is not None
    return S["theme"]


def current_theme() -> str:
    return _normalize(st.session_state.get("theme")) or "light"


def tokens(name: str | None = None) -> dict:
    return TOKENS[name or current_theme()]


def palette(name: str | None = None) -> dict:
    """Chart/mark colours for the active theme (pages use this instead of hex literals)."""
    t = tokens(name)
    return {k: t[k] for k in ("accent", "muted", "subtle", "highlight", "heat_low", "text", "text2", "success", "warning", "critical")}


def status_color(kind: str) -> str:
    return (_DARK_STATUS if current_theme() == "dark" else STATUS).get(kind, STATUS["neutral"])


def theme_toggle():
    """Light / Dark switch for the application header. Persisted via init_theme / persist_theme."""
    cur = current_theme()
    _, right = st.columns([7, 2])
    with right:
        st.segmented_control("Theme", [LABELS[t] for t in THEMES], default=LABELS[cur], key="theme_pick",
                             label_visibility="collapsed", help="Switch between the light and dark appearance.")
    try:
        st.query_params["theme"] = cur
    except Exception:       # query params are a convenience; never block the page
        pass


def persist_theme():
    """Remember the choice across browser refreshes. An explicit choice is saved; otherwise a saved choice is restored
    (one reload with ?theme=). The script is a zero-height component and touches nothing else."""
    import json as _json
    S = st.session_state
    script = """<script>
    (function () {
      var cur = %s, explicit = %s, key = "ws-theme";
      try {
        if (explicit) { window.localStorage.setItem(key, cur); return; }
        var saved = window.localStorage.getItem(key);
        if (saved && saved !== cur && (saved === "light" || saved === "dark") && !window.parent.__wsThemeTried) {
          window.parent.__wsThemeTried = true;          // once per page load: never loops
          var label = saved === "dark" ? "Dark" : "Light";
          window.setTimeout(function () {                // press the app's own toggle: the normal path, no navigation needed
            var btns = window.parent.document.querySelectorAll('[data-testid="stButtonGroup"] button');
            for (var i = 0; i < btns.length; i++) { if (btns[i].textContent.indexOf(label) >= 0) { btns[i].click(); break; } }
          }, 400);
        }
      } catch (e) { /* storage unavailable: the session still keeps the choice */ }
    })();
    </script>""" % (_json.dumps(current_theme()), "true" if S.get("theme_explicit") else "false")
    st.iframe(script, height=1)       # trusted static script built here (no user input); st.components.v1.html is deprecated


# ------------------------------------------------------------------------------------------------ plotly
_TEMPLATES: dict = {}


def _template(name: str) -> go.layout.Template:
    if name not in _TEMPLATES:
        t = TOKENS[name]
        tmpl = go.layout.Template()
        ax = dict(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["axis"], showgrid=True, ticks="outside",
                  tickcolor=t["axis"], tickfont=dict(color=t["text"]), title=dict(font=dict(color=t["text"])))
        tmpl.layout = go.Layout(
            paper_bgcolor=t["chart_bg"], plot_bgcolor=t["chart_bg"],
            font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=t["text"], size=12.5),
            colorway=t["categorical"], xaxis=ax, yaxis=ax,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(color=t["text"])),
            margin=dict(l=48, r=16, t=28, b=40),
            hoverlabel=dict(bgcolor=t["hover_bg"], bordercolor=t["axis"], font=dict(color=t["text"], size=12)),
            annotationdefaults=dict(font=dict(color=t["text"])),
        )
        _TEMPLATES[name] = tmpl
    return _TEMPLATES[name]


def styled_figure(fig: go.Figure, height: int = 340, name: str | None = None) -> go.Figure:
    """Apply the ACTIVE theme (or the named one) to a chart: axes, labels, legend, gridlines, tooltips, annotations."""
    name = name or current_theme()
    t = tokens(name)
    fig.update_layout(template=_template(name), height=height)
    # Streamlit's frontend fills any layout property the FIGURE leaves undefined with its own (light) defaults, and a
    # template does not count as defined, so the tokens are also set explicitly on the layout.
    axis = dict(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["axis"], tickcolor=t["axis"],
                tickfont=dict(color=t["text"]), title=dict(font=dict(color=t["text"])))
    fig.update_layout(paper_bgcolor=t["chart_bg"], plot_bgcolor=t["chart_bg"], font=dict(color=t["text"]),
                      legend=dict(font=dict(color=t["text"])), hoverlabel=dict(bgcolor=t["hover_bg"], font=dict(color=t["text"])))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def chart(fig: go.Figure, height: int = 340, container=None, **kwargs):
    """The one way pages draw a Plotly chart: themed figure + `theme=None`, because Streamlit's own chart theme would otherwise
    repaint the background and text of the figure (white in dark mode) and override the tokens."""
    kwargs.setdefault("width", "stretch")
    return (container or st).plotly_chart(styled_figure(fig, height), theme=None, **kwargs)


# ------------------------------------------------------------------------------------------------ css
_BASE_CSS = """
.block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 1200px; }
.ws-title { font-size: 1.35rem; font-weight: 650; margin: 0 0 0.1rem 0; letter-spacing: -0.01em; color: var(--ws-text); }
.ws-subtitle { font-size: 0.86rem; color: var(--ws-text2); margin: 0 0 1.1rem 0; }
.ws-section { font-size: 0.72rem; font-weight: 650; text-transform: uppercase; letter-spacing: 0.06em;
              color: var(--ws-text2); margin: 1.6rem 0 0.5rem 0; }
h1, h2, h3 { letter-spacing: -0.01em; }
h2 { font-size: 1.05rem !important; margin-top: 1.4rem !important; }
h3 { font-size: 0.92rem !important; }
div[data-testid="stMetric"] { background: var(--ws-surface); border: 1px solid var(--ws-border); border-radius: 8px;
                              padding: 0.7rem 0.9rem 0.6rem 0.9rem; }
div[data-testid="stMetricLabel"] { font-size: 0.72rem; }
div[data-testid="stMetricLabel"] p { white-space: normal; overflow: visible; text-overflow: clip; }
div[data-testid="stMetricValue"] { font-size: 1.35rem; }
@media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
.ws-pill { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.78rem; font-weight: 550;
           padding: 0.15rem 0.6rem; border-radius: 999px; }
.ws-pill-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
button[kind="primary"] { border-radius: 6px; }
button[data-baseweb="tab"] { font-size: 0.86rem; padding-top: 0.4rem; padding-bottom: 0.4rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
section[data-testid="stSidebar"] { border-right: 1px solid var(--ws-border); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--ws-text2); }
/* visible keyboard focus everywhere */
button:focus-visible, [role="button"]:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible,
[data-baseweb="select"]:focus-within, summary:focus-visible { outline: 2px solid var(--ws-accent) !important; outline-offset: 2px; }

/* ---------- result popup and cards ---------- */
[data-testid="stDialog"] > div { width: min(44rem, calc(100vw - 1.5rem)) !important; max-width: calc(100vw - 1.5rem) !important; }
[role="dialog"] { width: 100% !important; max-width: 100% !important; }
.ws-result { text-align: center; padding: 0.2rem 0.2rem 0.4rem 0.2rem; }
.ws-result-kicker { font-size: 0.72rem; font-weight: 650; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ws-text2); }
.ws-result-class { font-size: clamp(1.9rem, 6vw, 2.8rem); font-weight: 700; letter-spacing: -0.02em; line-height: 1.1;
                   margin: 0.35rem 0 0.1rem 0; color: var(--ws-text); }
.ws-result-model { font-size: 0.82rem; color: var(--ws-text2); margin-bottom: 0.9rem; }
.ws-result-label { font-size: 0.78rem; font-weight: 600; color: var(--ws-text2); margin-top: 0.7rem; }
.ws-result-value { font-size: 1.6rem; font-weight: 650; color: var(--ws-text); }
.ws-meter { position: relative; height: 12px; border-radius: 999px; background: var(--ws-surface);
            border: 1px solid var(--ws-border); margin: 0.35rem auto 0.2rem auto; max-width: 28rem; overflow: visible; }
.ws-meter > .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: var(--ws-accent); }
.ws-meter > .tick { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ws-text); opacity: 0.7; }
.ws-meter-note { font-size: 0.72rem; color: var(--ws-text2); }
.ws-agree { font-size: 1.05rem; font-weight: 600; color: var(--ws-text); }
.ws-divider { border: 0; border-top: 1px solid var(--ws-border); margin: 0.9rem 0; }
.ws-explain { text-align: left; font-size: 0.93rem; line-height: 1.5; color: var(--ws-text); }
.ws-explain b { font-weight: 650; }
.ws-note { text-align: left; font-size: 0.82rem; color: var(--ws-text2); border-left: 3px solid var(--ws-border);
           padding-left: 0.7rem; margin-top: 0.6rem; }
.ws-perf-head { font-size: 0.72rem; font-weight: 650; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ws-text2);
                margin: 0.9rem 0 0.2rem 0; text-align: left; }
.ws-latest { border: 1px solid var(--ws-border); background: var(--ws-surface); border-radius: 10px; padding: 0.6rem 0.9rem; }
.ws-latest b { font-weight: 650; }
.ws-latest .meta { color: var(--ws-text2); font-size: 0.8rem; }
.ws-card { border: 1px solid var(--ws-border); background: var(--ws-surface); border-radius: 10px; padding: 0.8rem 1rem; }
.ws-card dt { font-size: 0.74rem; color: var(--ws-text2); margin-top: 0.5rem; }
.ws-card dd { margin: 0; font-size: 1rem; font-weight: 600; color: var(--ws-text); }
"""

# Streamlit's native widgets read their colours from compiled theme values, so the dark theme overrides them here,
# using the same tokens (this block is only emitted when the dark theme is active).
_DARK_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"] { background: var(--ws-bg) !important; color: var(--ws-text) !important; }
[data-testid="stHeader"] { background: var(--ws-bg) !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: var(--ws-surface) !important; }
:is(.stApp, [data-testid="stDialog"]) h1, :is(.stApp, [data-testid="stDialog"]) h2, :is(.stApp, [data-testid="stDialog"]) h3, :is(.stApp, [data-testid="stDialog"]) h4, :is(.stApp, [data-testid="stDialog"]) h5, :is(.stApp, [data-testid="stDialog"]) h6, :is(.stApp, [data-testid="stDialog"]) p, :is(.stApp, [data-testid="stDialog"]) li, :is(.stApp, [data-testid="stDialog"]) label, :is(.stApp, [data-testid="stDialog"]) legend,
:is(.stApp, [data-testid="stDialog"]) [data-testid="stMarkdownContainer"], :is(.stApp, [data-testid="stDialog"]) [data-testid="stWidgetLabel"], :is(.stApp, [data-testid="stDialog"]) [data-testid="stMetricValue"],
:is(.stApp, [data-testid="stDialog"]) [data-testid="stMetricLabel"], :is(.stApp, [data-testid="stDialog"]) summary, :is(.stApp, [data-testid="stDialog"]) [data-testid="stExpander"] details summary p { color: var(--ws-text) !important; }
:is(.stApp, [data-testid="stDialog"]) [data-testid="stCaptionContainer"], :is(.stApp, [data-testid="stDialog"]) .stCaption, :is(.stApp, [data-testid="stDialog"]) small { color: var(--ws-text2) !important; }
:is(.stApp, [data-testid="stDialog"]) a { color: var(--ws-info) !important; }
/* inputs, selects, uploaders */
:is(.stApp, [data-testid="stDialog"]) input, :is(.stApp, [data-testid="stDialog"]) textarea, :is(.stApp, [data-testid="stDialog"]) [data-baseweb="input"], :is(.stApp, [data-testid="stDialog"]) [data-baseweb="base-input"], :is(.stApp, [data-testid="stDialog"]) [data-baseweb="textarea"],
:is(.stApp, [data-testid="stDialog"]) [data-baseweb="select"] > div { background: var(--ws-input) !important; color: var(--ws-text) !important; border-color: var(--ws-border) !important; }
:is(.stApp, [data-testid="stDialog"]) [data-baseweb="select"] svg, :is(.stApp, [data-testid="stDialog"]) [data-baseweb="input"] svg { fill: var(--ws-text2) !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul,
[data-baseweb="popover"] li, [data-baseweb="calendar"] { background: var(--ws-elevated) !important; color: var(--ws-text) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover { background: var(--ws-surface) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--ws-surface) !important; border-color: var(--ws-border) !important; color: var(--ws-text) !important; }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: var(--ws-text2) !important; }
/* buttons (primary keeps the accent) */
:is(.stApp, [data-testid="stDialog"]) [data-testid^="stBaseButton-secondary"], :is(.stApp, [data-testid="stDialog"]) [data-testid="stBaseButton-minimal"], :is(.stApp, [data-testid="stDialog"]) .stDownloadButton button,
:is(.stApp, [data-testid="stDialog"]) [data-testid^="stBaseButton-segmented_control"], :is(.stApp, [data-testid="stDialog"]) [data-testid="stBaseButton-pills"] {
  background: var(--ws-elevated) !important; color: var(--ws-text) !important; border: 1px solid var(--ws-border) !important; }
:is(.stApp, [data-testid="stDialog"]) [data-testid="stBaseButton-segmented_controlActive"], :is(.stApp, [data-testid="stDialog"]) [data-testid="stBaseButton-pillsActive"] {
  background: var(--ws-accent) !important; color: var(--ws-accent-text) !important; border-color: var(--ws-accent) !important; }
:is(.stApp, [data-testid="stDialog"]) [data-testid="stBaseButton-primary"] { background: var(--ws-accent) !important; color: var(--ws-accent-text) !important; border: 1px solid var(--ws-accent) !important; }
:is(.stApp, [data-testid="stDialog"]) [data-testid="stBaseButton-primary"] p, :is(.stApp, [data-testid="stDialog"]) [data-testid="stBaseButton-segmented_controlActive"] p { color: var(--ws-accent-text) !important; }
:is(.stApp, [data-testid="stDialog"]) button:disabled { opacity: 0.5 !important; }
/* segmented controls (no test-id in this Streamlit version: identified by aria-checked) */
[data-testid="stButtonGroup"] button { background: var(--ws-elevated) !important; color: var(--ws-text) !important; border: 1px solid var(--ws-border) !important; }
[data-testid="stButtonGroup"] button * { color: inherit !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] { background: var(--ws-accent) !important; color: var(--ws-accent-text) !important; border-color: var(--ws-accent) !important; }
/* select boxes (react-aria markup in this Streamlit version: a role="group" wrapper with an arrow button, options in a listbox) */
:is(.stApp, [data-testid="stDialog"]) :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) [role="group"],
:is(.stApp, [data-testid="stDialog"]) :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) [role="group"] button {
  background-color: var(--ws-input) !important; color: var(--ws-text) !important; border-color: var(--ws-border) !important; }
:is(.stApp, [data-testid="stDialog"]) :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) [role="group"] svg { fill: var(--ws-text2) !important; color: var(--ws-text2) !important; }
[role="listbox"], [role="listbox"] [role="option"], [data-rac][data-trigger], [data-rac][data-trigger] > div { background-color: var(--ws-elevated) !important; color: var(--ws-text) !important; }
[role="listbox"] [role="option"][data-focused="true"], [role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][aria-selected="true"] { background-color: var(--ws-surface) !important; }
/* legacy markup: every nested layer takes the input colour (the arrow cell has its own background) */
:is(.stApp, [data-testid="stDialog"]) [data-baseweb="select"] div { background-color: var(--ws-input) !important; color: var(--ws-text) !important; }
:is(.stApp, [data-testid="stDialog"]) [data-baseweb="select"] svg { fill: var(--ws-text2) !important; }
/* tabs, expanders, status, alerts, code, json */
:is(.stApp, [data-testid="stDialog"]) button[data-baseweb="tab"] { color: var(--ws-text2) !important; }
:is(.stApp, [data-testid="stDialog"]) button[data-baseweb="tab"][aria-selected="true"] { color: var(--ws-text) !important; }
:is(.stApp, [data-testid="stDialog"]) [data-baseweb="tab-border"], :is(.stApp, [data-testid="stDialog"]) [data-baseweb="tab-list"] { background-color: transparent !important; }
[data-testid="stExpander"] details, [data-testid="stStatus"] details { background: var(--ws-surface) !important; border-color: var(--ws-border) !important; }
[data-testid="stAlert"] > div { background: var(--ws-elevated) !important; color: var(--ws-text) !important; border: 1px solid var(--ws-border); }
[data-testid="stAlert"] p { color: var(--ws-text) !important; }
[data-testid="stCode"], [data-testid="stCode"] pre, :is(.stApp, [data-testid="stDialog"]) pre, :is(.stApp, [data-testid="stDialog"]) code { background: var(--ws-surface) !important; color: var(--ws-text) !important; }
[data-testid="stJson"], [data-testid="stJson"] * { background: var(--ws-surface) !important; color: var(--ws-text) !important; }
/* tooltips and the result dialog */
[data-testid="stTooltipContent"], [role="tooltip"], [data-baseweb="tooltip"] > div { background: var(--ws-elevated) !important; color: var(--ws-text) !important; }
[data-testid="stDialog"] > div, [role="dialog"], [role="dialog"] > div { background: var(--ws-elevated) !important; color: var(--ws-text) !important; border: 1px solid var(--ws-border); }
[role="dialog"] h1, [role="dialog"] h2, [role="dialog"] h3, [role="dialog"] p, [role="dialog"] label, [role="dialog"] [data-testid="stMarkdownContainer"] { color: var(--ws-text) !important; }
/* canvas tables cannot be styled by CSS variables: invert them into a dark grid */
[data-testid="stDataFrame"], [data-testid="stTable"] { filter: invert(0.92) hue-rotate(180deg); border-radius: 6px; }
/* static PNG figures were drawn on white: keep them readable on a light card */
[data-testid="stImage"] img { background: #ffffff; padding: 6px; border-radius: 8px; }
hr { border-color: var(--ws-border) !important; }
"""


def _root_vars(t: dict) -> str:
    names = ["bg", "surface", "elevated", "text", "text2", "border", "input", "accent", "accent_text", "info", "success",
             "warning", "critical", "grid", "axis", "muted", "subtle", "highlight", "heat_low", "shadow"]
    return ":root { " + " ".join(f"--ws-{n.replace('_', '-')}: {t[n]};" for n in names) + " }"


def inject_css():
    name = current_theme()
    css = _root_vars(TOKENS[name]) + _BASE_CSS + (_DARK_CSS if name == "dark" else "")
    st.markdown(f"<style data-ws-theme='{name}'>{css}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ small components
def page_header(title: str, subtitle: str | None = None):
    st.markdown(f'<div class="ws-title">{_html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ws-subtitle">{_html.escape(subtitle)}</div>', unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="ws-section">{_html.escape(text)}</div>', unsafe_allow_html=True)


def status_pill(text: str, kind: str = "neutral"):
    color = status_color(kind)
    st.markdown(
        f'<span class="ws-pill" style="background:{color}26; color:{color};">'
        f'<span class="ws-pill-dot" style="background:{color};"></span>{_html.escape(text)}</span>',
        unsafe_allow_html=True,
    )
