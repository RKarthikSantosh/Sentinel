import sys
sys.path.append('src')
import pandas as pd
from predict import predict_attack

df = pd.read_csv('data/raw/KDDTest+.txt', header=None)
print('First 10 rows — actual vs predicted:\n')
for i, (_, row) in enumerate(df.head(10).iterrows()):
    actual = row[41]
    sample = pd.DataFrame([row.values])
    result = predict_attack(sample)
    predicted = result['attack_name']
    risk = result['risk_score']
    threat = result['threat_level']
    fp = " <-- FALSE POSITIVE" if actual == 'normal' and predicted != 'normal' else ""
    fn = " <-- MISSED ATTACK" if actual != 'normal' and predicted == 'normal' else ""
    print(f"Row {i}: actual={actual:<15} predicted={predicted:<15} risk={risk:>3}  threat={threat}{fp}{fn}")
