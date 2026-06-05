def calculate_risk(confidence):

    risk_score = int(confidence * 100)

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

    confidence = 0.96

    score, level = calculate_risk(
        confidence
    )

    print(
        f"Risk Score: {score}"
    )

    print(
        f"Threat Level: {level}"
    )