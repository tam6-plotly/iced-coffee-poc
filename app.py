"""Explore POC interaction summaries with Dash AI Analyst.

  python app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import dash_ag_grid as dag
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from dash_ai_analyst import AiAnalyst, ChatStyle, register_data

ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "poc_interactions.parquet"

ACCOUNT_NAMES = {
    "0011U000003MS4vQAG": "AbbVie",
    "0011U000003MSKYQA4": "Fidelity Investments",
    "0011U000003MRydQAG": "Apple",
    "0011U000003MRx3QAG": "Bank of America, N.A.",
    "0011U000003MSAYQA4": "Cummins",
    "0011U000009OQbFQAW": "CIBC",
    "001I90000079KB5IAM": "Jisc",
    "001OL00000R6gzbYAB": "Siemens Energy",
    "0011U00001tBGJxQAO": "Immersive Labs Corporation",
    "0011U00001M4VO8QAN": "Kansas City Chiefs",
}

ACCOUNT_ORDER = list(ACCOUNT_NAMES)

LIST_COLUMNS = (
    "feature_mentions",
    "key_takeaways",
    "next_steps",
    "attendees",
    "mentioned",
)

GRID_FIELDS = [
    "timestamp",
    "timestamp_latest",
    "source_object",
    "source_type",
    "summary",
    "feature_mentions",
    "key_takeaways",
    "next_steps",
    "attendees",
]

COLUMN_DEFS = [
    {"field": "timestamp", "headerName": "Timestamp", "sort": "desc", "width": 160, "wrapText": False, "autoHeight": False},
    {"field": "timestamp_latest", "headerName": "Latest", "width": 160, "wrapText": False, "autoHeight": False},
    {"field": "source_object", "headerName": "Source", "width": 130},
    {"field": "source_type", "headerName": "Type", "width": 110},
    {"field": "summary", "headerName": "Summary", "flex": 2, "minWidth": 280},
    {"field": "feature_mentions", "headerName": "Features", "width": 160},
    {"field": "key_takeaways", "headerName": "Takeaways", "flex": 1.5, "minWidth": 220},
    {"field": "next_steps", "headerName": "Next steps", "flex": 1, "minWidth": 180},
    {"field": "attendees", "headerName": "Attendees", "minWidth": 180},
]


def _join_list(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value
    items = list(value)
    parts = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("email") or ""
            role = item.get("role")
            parts.append(f"{name} ({role})" if name and role else str(name))
        elif item is not None:
            parts.append(str(item))
    return "; ".join(p for p in parts if p)


def _fmt_ts(value) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d %H:%M")


def load_interactions() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["account_name"] = df["account_id"].map(ACCOUNT_NAMES).fillna(df["account_id"])
    for col in LIST_COLUMNS:
        df[col] = df[col].map(_join_list)
    return df.sort_values(["timestamp", "timestamp_latest"], ascending=False, ignore_index=True)


INTERACTIONS = load_interactions()

register_data(
    INTERACTIONS,
    "poc_interactions",
    "Customer interaction summaries loaded from poc_interactions.parquet "
    "(Salesforce emails, tickets, and meetings). Columns: record_id, account_id, "
    "account_name, source_object, timestamp, timestamp_latest, source_type, summary, "
    "feature_mentions, key_takeaways, next_steps, attendees, mentioned. "
    "List fields are semicolon-separated text.",
)

present_ids = set(INTERACTIONS["account_id"].unique())
ACCOUNT_OPTIONS = [
    {"label": ACCOUNT_NAMES[account_id], "value": account_id}
    for account_id in ACCOUNT_ORDER
    if account_id in present_ids
]
DEFAULT_ACCOUNT = ACCOUNT_OPTIONS[0]["value"] if ACCOUNT_OPTIONS else None

app = Dash(__name__)
server = app.server
server.secret_key = os.environ.get("SECRET_KEY", "poc-app-dev")
app.title = "Customer interactions"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Customer interactions", style={"margin": "0 0 12px 0", "fontSize": "22px"}),
                dcc.Dropdown(
                    id="account-dropdown",
                    options=ACCOUNT_OPTIONS,
                    value=DEFAULT_ACCOUNT,
                    clearable=False,
                    style={"maxWidth": "420px"},
                ),
            ],
            style={"padding": "16px 20px", "borderBottom": "1px solid #e5e7eb", "flexShrink": 0},
        ),
        html.Div(
            [
                html.Div(
                    dag.AgGrid(
                        id="records-grid",
                        columnDefs=COLUMN_DEFS,
                        rowData=[],
                        defaultColDef={
                            "resizable": True,
                            "sortable": True,
                            "filter": True,
                            "wrapText": True,
                            "autoHeight": True,
                        },
                        dashGridOptions={"animateRows": False},
                        style={"height": "100%", "width": "100%"},
                    ),
                    style={"flex": "1", "minWidth": 0, "padding": "12px 16px", "overflow": "hidden"},
                ),
                html.Div(
                    AiAnalyst(
                        inline=True,
                        skip_global_data=True,
                        style=ChatStyle(
                            title="AI Analyst",
                            placeholder="Ask about these customer interactions…",
                        ),
                        system_prompt_appendix=(
                            "You are exploring Plotly customer interaction summaries from "
                            "poc_interactions.parquet (registered as poc_interactions). "
                            "Filter by account_name or account_id when the user names a customer. "
                            "Timestamps are timezone-aware datetimes."
                        ),
                    ),
                    style={
                        "width": "420px",
                        "flexShrink": 0,
                        "borderLeft": "1px solid #e5e7eb",
                        "height": "100%",
                    },
                ),
            ],
            style={"display": "flex", "alignItems": "stretch", "flex": "1", "minHeight": 0},
        ),
    ],
    style={
        "fontFamily": "system-ui, sans-serif",
        "height": "100vh",
        "margin": 0,
        "display": "flex",
        "flexDirection": "column",
    },
)


@app.callback(Output("records-grid", "rowData"), Input("account-dropdown", "value"))
def filter_grid(account_id: str | None):
    if not account_id:
        return []
    view = INTERACTIONS[INTERACTIONS["account_id"] == account_id].copy()
    view["timestamp"] = view["timestamp"].map(_fmt_ts)
    view["timestamp_latest"] = view["timestamp_latest"].map(_fmt_ts)
    return view[GRID_FIELDS].to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)
