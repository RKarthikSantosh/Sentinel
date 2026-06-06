def calculate_risk(confidence, margin=None):
    """
    Compute a risk score (0-100) and threat level.
    """
    if margin is not None:
        # Blend confidence (70 %) with margin (30 %)
        blended = 0.70 * confidence + 0.30 * margin
        risk_score = int(blended * 100)
    else:
        risk_score = int(confidence * 100)

    # Clamp to [0, 100]
    risk_score = max(0, min(100, risk_score))

    if risk_score <= 30:
        level = "Low"
    elif risk_score <= 60:
        level = "Medium"
    elif risk_score <= 85:
        level = "High"
    else:
        level = "Critical"

    return risk_score, level


if __name__ == "__main__":

    # Example: high confidence but low margin → lower risk
    for conf, mrg in [(0.96, 0.90), (0.96, 0.30), (0.60, 0.10)]:
        score, level = calculate_risk(conf, mrg)
        print(f"Confidence={conf}  Margin={mrg}  -> Risk={score}  Level={level}")