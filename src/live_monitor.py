import pandas as pd


def extract_features(packet):
    """
    Extract basic features from a Scapy packet and map them
    to the 41-feature format expected by the NSL-KDD model.
    Scapy is imported lazily so the module loads on Streamlit Cloud
    even when packet capture is unavailable.
    """
    from scapy.all import IP, TCP, UDP, ICMP  # lazy import

    if IP not in packet:
        return None

    # Initialize a row of zeros matching the KDD 41 features
    features = [0] * 41

    # 0 - duration
    features[0] = 0

    # 1 - protocol_type
    if TCP in packet:
        features[1] = 'tcp'
    elif UDP in packet:
        features[1] = 'udp'
    elif ICMP in packet:
        features[1] = 'icmp'
    else:
        return None

    # 2 - service
    port = packet[IP].dport if hasattr(packet[IP], 'dport') else 0
    if port == 80:
        features[2] = 'http'
    elif port == 443:
        features[2] = 'https'
    elif port == 21:
        features[2] = 'ftp'
    elif port == 22:
        features[2] = 'ssh'
    elif port == 53:
        features[2] = 'domain_u'
    else:
        features[2] = 'private'

    # 3 - flag (simulate a standard connection)
    features[3] = 'SF'

    # 4 - src_bytes
    features[4] = len(packet[IP].payload)

    # 5 - dst_bytes (0 for outgoing single packet)
    features[5] = 0

    df = pd.DataFrame([features])
    return {
        "src_ip": packet[IP].src,
        "dst_ip": packet[IP].dst,
        "protocol": features[1],
        "features_df": df
    }


def capture_packets_simulated(packet_count, df_mock):
    """
    Simulates packet capture by reading rows from a dataframe.
    Useful for cloud environments lacking Npcap or admin privileges.
    """
    import random
    import time

    captured_data = []
    sample_df = df_mock.sample(n=packet_count, replace=True)

    for _, row in sample_df.iterrows():
        time.sleep(random.uniform(0.01, 0.1))

        src_ip = f"192.168.1.{random.randint(2, 254)}"
        dst_ip = f"10.0.0.{random.randint(1, 100)}"

        protocol = str(row.iloc[1]) if isinstance(row.iloc[1], str) else "tcp"

        features_df = pd.DataFrame([row.values])

        data = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": protocol,
            "features_df": features_df
        }
        captured_data.append(data)

    return captured_data


def capture_packets(packet_count):
    """
    Captures a set number of live packets using Scapy.
    Requires Npcap (Windows) or libpcap (Linux/macOS) and elevated privileges.
    Scapy is imported lazily so module-level import never fails on Streamlit Cloud.
    """
    from scapy.all import sniff, conf  # lazy import

    captured_data = []

    def packet_handler(packet):
        data = extract_features(packet)
        if data is not None:
            captured_data.append(data)

    # L3socket avoids needing WinPcap/Npcap on some Windows setups
    sniff(prn=packet_handler, store=0, count=packet_count, L2socket=conf.L3socket)
    return captured_data
