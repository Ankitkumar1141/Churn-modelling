import json
import os
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIER_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")


@st.cache_data(ttl=60)
def get_meta():
    r = requests.get(f"{API_URL}/meta", timeout=5)
    r.raise_for_status()
    return r.json()


def gauge(score: float, meta: dict) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100,
        number={"suffix": "%"}, title={"text": "Churn risk score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#37474f"},
               "steps": [
                   {"range": [0, meta["medium_risk"] * 100], "color": "#c8e6c9"},
                   {"range": [meta["medium_risk"] * 100, meta["high_risk"] * 100], "color": "#fff3c4"},
                   {"range": [meta["high_risk"] * 100, 100], "color": "#ffcdd2"}]}))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def api_error(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        return resp.text
    if isinstance(detail, list):  # pydantic validation errors
        return "\n".join(f"- {'.'.join(map(str, d['loc'][1:]))}: {d['msg']}" for d in detail[:10])
    return str(detail)


st.title("📉 Customer Churn Predictor")

try:
    meta = get_meta()
except Exception:
    st.error(f"Cannot reach the API at {API_URL}. Start it with "
             "`uvicorn app:app --port 8000`.")
    st.stop()

tab_single, tab_batch = st.tabs(["Single customer", "Batch (CSV)"])

# ---------------- single ----------------
with tab_single:
    with st.form("customer"):
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", meta["genders"], format_func=str.title)
        geography = c2.selectbox("Geography", meta["geographies"], format_func=str.title)
        age = c1.number_input("Age", 18, 100, 40)
        tenure = c2.number_input("Tenure (years)", 0, 50, 5)
        salary = c1.number_input("Salary", min_value=0.0, value=50000.0, step=1000.0)
        num_products = c2.number_input("Number of products", 1, 10, 1)
        balance_known = st.checkbox("Balance is known", value=True)
        balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0,
                                  disabled=not balance_known)
        c3, c4 = st.columns(2)
        has_cc = c3.checkbox("Has credit card", value=True)
        is_active = c4.checkbox("Is active member", value=True)
        go_btn = st.form_submit_button("Predict", type="primary")

    if go_btn:
        payload = {"gender": gender, "age": int(age), "salary": salary,
                   "geography": geography, "tenure": int(tenure),
                   "balance": balance if balance_known else None,
                   "num_products": int(num_products),
                   "has_credit_card": int(has_cc), "is_active": int(is_active)}
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
        else:
            if resp.ok:
                res = resp.json()
                st.plotly_chart(gauge(res["churn_score"], meta), use_container_width=True)
                tier = res["risk_tier"]
                st.markdown(f"### Risk tier: <span style='color:{TIER_COLORS[tier]}'>{tier}</span>",
                            unsafe_allow_html=True)
                st.write("Predicted to churn" if res["churn_prediction"] else "Predicted to stay")
                st.caption("The model was trained on a class-balanced sample, so the score "
                           "ranks risk but overstates the real-world churn probability.")
            else:
                st.error(api_error(resp))

# ---------------- batch ----------------
with tab_batch:
    st.write("Upload a CSV with columns: " + ", ".join(f"`{f}`" for f in meta["features"])
             + ". Extra columns (e.g. `churned`) are ignored.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        missing = [f for f in meta["features"] if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif len(df) > meta["max_batch_size"]:
            st.error(f"Max {meta['max_batch_size']} rows per upload.")
        else:
            st.dataframe(df.head(), use_container_width=True)
            if st.button("Run predictions", type="primary"):
                feats = df[meta["features"]]
                # to_json turns NaN into null, which the API treats as a missing balance
                records = json.loads(feats.to_json(orient="records"))
                with st.spinner("Scoring..."):
                    resp = requests.post(f"{API_URL}/predict/batch",
                                         json={"customers": records}, timeout=120)
                if resp.ok:
                    preds = pd.DataFrame(resp.json()["predictions"])
                    out = pd.concat([df.reset_index(drop=True), preds], axis=1)
                    st.success(f"Scored {len(out)} customers")
                    st.bar_chart(out["risk_tier"].value_counts()
                                 .reindex(["Low", "Medium", "High"]).fillna(0))
                    st.dataframe(out, use_container_width=True)
                    st.download_button("Download results", out.to_csv(index=False),
                                       "churn_predictions.csv", "text/csv")
                else:
                    st.error(api_error(resp))
