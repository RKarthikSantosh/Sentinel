import streamlit as st
import pandas as pd
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.abspath('src'))
from predict import predict_attack
from report_generator import generate_report, report_to_pdf_bytes
from live_monitor import capture_packets, capture_packets_simulated

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
            df = pd.read_csv(uploaded_file, header=None)
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
    st.header("📡 Mode 2: Live Firewall / Packet Monitoring")
    st.markdown("Captures live network packets passing through your network interface, extracts features, and runs them against the ML Model.")
    
    packet_count = st.slider("Number of packets to capture per batch", min_value=10, max_value=500, value=50)
    
    if st.button("Start Live Capture"):
        captured_data = []
        with st.spinner(f"Capturing {packet_count} packets..."):
            try:
                captured_data = capture_packets(packet_count)
                st.success(f"Captured {len(captured_data)} valid IP packets natively.")
            except Exception as e:
                # Catch Scapy Permission/Npcap errors on Windows
                st.warning("⚠️ Native Scapy capture requires Npcap or Administrator privileges on Windows. Switching to Live Simulation Mode using historical logs...")
                try:
                    mock_df = pd.read_csv("data/raw/KDDTest+.txt", header=None)
                    captured_data = capture_packets_simulated(packet_count, mock_df)
                    st.success(f"Simulated {len(captured_data)} live packets from firewall logs.")
                except Exception as ex:
                    st.error(f"Failed to run simulation: {ex}")
        
        if not captured_data:
            st.stop()

        # Analyze packets
        threats_found = []
        normal_count = 0
        
        progress_text = "Analyzing packets..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, data in enumerate(captured_data):
            my_bar.progress((i + 1) / len(captured_data), text=progress_text)
            
            try:
                result = predict_attack(data["features_df"])
                if result['attack_name'] != 'normal':
                    threats_found.append({
                        "Source IP": data["src_ip"],
                        "Dest IP": data["dst_ip"],
                        "Protocol": data["protocol"].upper(),
                        "Attack Type": result['attack_name'],
                        "Confidence": result['confidence'],
                        "Risk": result['risk_score']
                    })
                else:
                    normal_count += 1
            except Exception as e:
                pass # skip prediction errors on unfamiliar data
        
        st.write(f"**Normal Packets:** {normal_count}")
        st.write(f"**Threats Detected:** {len(threats_found)}")
        
        if threats_found:
            st.error("⚠️ Threats Detected in Live Traffic!")
            threats_df = pd.DataFrame(threats_found)
            st.dataframe(threats_df)
            
            # Generate report for the most severe threat
            highest_risk = max(threats_found, key=lambda x: x['Risk'])
            
            st.subheader(f"AI Report for Severe Threat: {highest_risk['Attack Type']}")
            with st.spinner("Generating Report..."):
                report = generate_report(
                    attack_name=highest_risk['Attack Type'],
                    confidence=highest_risk['Confidence'],
                    risk_score=highest_risk['Risk'],
                    threat_level="Critical" if highest_risk['Risk'] > 75 else "High"
                )
            display_incident_report(report, highest_risk["Attack Type"])
        else:
            st.success("✅ Architecture is secure. No intrusions detected in this capture batch.")
