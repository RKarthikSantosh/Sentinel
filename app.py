import streamlit as st
import pandas as pd
import sys
import os

# Anchor src/ path to app.py location — works regardless of CWD on Streamlit Cloud
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, 'src'))

from predict import predict_attack
from report_generator import generate_report, report_to_pdf_bytes
from data_preprocessing import FEATURE_NAMES

st.set_page_config(page_title="Intrusion Detection SOC Dashboard", layout="wide")


def display_incident_report(report, attack_name):
    st.subheader("Incident Report")
    st.markdown(report)
    st.download_button(
        label="Download Report (PDF)",
        data=report_to_pdf_bytes(report),
        file_name=f"security_incident_report_{attack_name}.pdf",
        mime="application/pdf",
        type="primary",
    )

st.title("🛡️ AI-Driven Intrusion Detection Dashboard")
st.markdown("Dual-mode AI-powered Intrusion Detection System supporting both historical log analysis and real-time network traffic monitoring.")

mode = st.sidebar.radio("Select Mode", ["Mode 1: Historical CSV Log Analysis", "Mode 2: Real-Time Network Monitoring"])

if mode == "Mode 1: Historical CSV Log Analysis":
    st.header("📂 Mode 1: Historical CSV Upload")
    
    uploaded_file = st.file_uploader("Upload Network Traffic CSV (e.g. KDDTest+.txt)", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=FEATURE_NAMES)
            st.success(f"Successfully loaded {len(df)} rows!")
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            row_index = st.number_input("Select row to analyze", min_value=0, max_value=len(df)-1, value=0, step=1)
            if st.button("Analyze Row"):
                sample = df.iloc[[row_index]]
                with st.spinner("Analyzing with Machine Learning Model..."):
                    result = predict_attack(sample)
                
                cols = st.columns(3)
                cols[0].metric("Attack Type", result['attack_name'].upper())
                cols[1].metric("Confidence", f"{result['confidence']}%")
                cols[2].metric("Risk Score", f"{result['risk_score']}/100")
                
                st.warning(f"Threat Level: **{result['threat_level']}**")
                
                if result['attack_name'] != 'normal':
                    with st.spinner("Generating Security Report..."):
                        report = generate_report(
                            attack_name=result['attack_name'],
                            confidence=result['confidence'],
                            risk_score=result['risk_score'],
                            threat_level=result['threat_level']
                        )
                    display_incident_report(report, result["attack_name"])
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    import time
    import random

    st.header("📡 Mode 2: Real-Time Firewall Log Monitoring")
    st.markdown(
        "Streams firewall log entries through the **Random Forest IDS** in real-time, "
        "flagging threats as they arrive."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    log_speed = col_cfg1.slider("Log ingestion speed (entries/sec)", 1, 20, 5)
    log_count = col_cfg2.slider("Number of log entries to process", 10, 300, 60)

    if st.button("▶ Start Monitoring", type="primary"):

        # Load firewall log source (KDDTest+ as stand-in for real firewall logs)
        _data_path = os.path.join(_APP_DIR, "data", "raw", "KDDTest+.txt")
        try:
            log_df = pd.read_csv(_data_path, header=None)
        except FileNotFoundError:
            st.error(f"Firewall log source not found at {_data_path}")
            st.stop()

        sample_df = log_df.sample(n=min(log_count, len(log_df)), replace=True).reset_index(drop=True)

        # --- Live stat placeholders ---
        st.markdown("---")
        stat_cols = st.columns(4)
        ph_total    = stat_cols[0].empty()
        ph_threats  = stat_cols[1].empty()
        ph_normal   = stat_cols[2].empty()
        ph_risk     = stat_cols[3].empty()

        st.markdown("#### 🔴 Live Event Feed")
        ph_feed     = st.empty()   # live-updating table
        ph_alert    = st.empty()   # banner for critical hits

        # --- State ---
        events      = []
        threats     = []
        normal_count = 0
        delay = 1.0 / log_speed

        protocols = ["TCP", "UDP", "ICMP"]

        for i, (_, row) in enumerate(sample_df.iterrows()):
            features_df = pd.DataFrame([row.values])
            src_ip = f"192.168.{random.randint(0,5)}.{random.randint(1,254)}"
            dst_ip = f"10.0.0.{random.randint(1,50)}"
            timestamp = pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3]

            try:
                result = predict_attack(features_df)
            except Exception:
                time.sleep(delay)
                continue

            is_attack = result["attack_name"] != "normal"
            status_icon = "🔴 THREAT" if is_attack else "🟢 NORMAL"

            event = {
                "Time": timestamp,
                "Src IP": src_ip,
                "Dst IP": dst_ip,
                "IDS Decision": status_icon,
                "Attack Type": result["attack_name"].upper() if is_attack else "—",
                "Confidence": f"{result['confidence']}%",
                "Risk Score": result["risk_score"] if is_attack else 0,
            }
            events.append(event)

            if is_attack:
                threats.append({**result, "src_ip": src_ip, "dst_ip": dst_ip})
                ph_alert.error(
                    f"⚠️  **THREAT DETECTED** — `{src_ip}` → `{dst_ip}` "
                    f"| **{result['attack_name'].upper()}** "
                    f"| Confidence: {result['confidence']}% "
                    f"| Risk: {result['risk_score']}/100"
                )
            else:
                normal_count += 1
                ph_alert.empty()

            # Update stats
            ph_total.metric("📋 Logs Analyzed", i + 1)
            ph_threats.metric("🔴 Threats Found", len(threats))
            ph_normal.metric("🟢 Normal Traffic", normal_count)
            top_risk = max((t["risk_score"] for t in threats), default=0)
            ph_risk.metric("🔥 Highest Risk", f"{top_risk}/100")

            # Update live feed (show last 20 events, newest first)
            feed_df = pd.DataFrame(events[::-1][:20])
            ph_feed.dataframe(feed_df, use_container_width=True, hide_index=True)

            time.sleep(delay)

        # --- Final summary ---
        st.markdown("---")
        st.subheader("📊 Monitoring Complete — Summary")
        fin_cols = st.columns(3)
        fin_cols[0].metric("Total Logs Processed", len(events))
        fin_cols[1].metric("Threats Detected", len(threats))
        fin_cols[2].metric("Normal Traffic", normal_count)

        if threats:
            st.error(f"⚠️ {len(threats)} threat(s) detected during this monitoring window.")

            # Report for highest-risk threat
            highest = max(threats, key=lambda x: x["risk_score"])
            st.subheader(f"🗂 Incident Report — {highest['attack_name'].upper()}")
            report = generate_report(
                attack_name=highest["attack_name"],
                confidence=highest["confidence"],
                risk_score=highest["risk_score"],
                threat_level=highest["threat_level"],
            )
            display_incident_report(report, highest["attack_name"])
        else:
            st.success("✅ No threats detected. Network appears clean.")

