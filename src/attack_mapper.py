ATTACK_DESCRIPTIONS = {

    "normal":
        "Normal Network Activity",

    "neptune":
        "Denial of Service (DoS) Attack",

    "smurf":
        "Distributed Denial of Service (DDoS) Attack",

    "satan":
        "Port Scanning Activity",

    "ipsweep":
        "Network Reconnaissance Attack",

    "portsweep":
        "Port Scanning Activity",

    "nmap":
        "Network Mapping Activity"
}
def get_attack_description(attack_name):

    return ATTACK_DESCRIPTIONS.get(
        attack_name,
        attack_name
    )