"""
============================================================================
TEST SCRIPT - test_endpoint.py
============================================================================
"""

from sagemaker.sklearn import SKLearnPredictor
from sagemaker import Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

ENDPOINT_NAME = "churn-prediction-endpoint-v2"

print("=" * 70)
print("TESTING CHURN PREDICTION ENDPOINT")
print("=" * 70)

session = Session()

predictor = SKLearnPredictor(
    endpoint_name=ENDPOINT_NAME,
    sagemaker_session=session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

print(f"✓ Connected to endpoint: {ENDPOINT_NAME}\n")

# -------------------------------------------------------------------------
# TEST CASES — MUST MATCH FEATURE_ORDER IN inference.py
# -------------------------------------------------------------------------
test_cases = [
    {
        "name": "High Risk Customer",
        "description": "Young, month-to-month, high charges, no support",
        "data": {
            "gender_male": 0,
            "age": 59,
            "monthly_charges": 89.15,
            "total_charges": 9950.00,
            "tenure": 4,
            "contract_month_to_month": 1,
            "contract_two_year": 0,
            "internet_dsl": 0,
            "tech_support_yes": 0,
            "streaming_tv_yes": 1,
            "payment_electronic_check": 1
        }
    },
    {
        "name": "Low Risk Customer",
        "description": "Senior, two-year contract, low charges, has support",
        "data": {
            "gender_male": 1,
            "age": 52,
            "monthly_charges": 144.85,
            "total_charges": 3200.00,
            "tenure": 71,
            "contract_month_to_month": 0,
            "contract_two_year": 1,
            "internet_dsl": 1,
            "tech_support_yes": 1,
            "streaming_tv_yes": 0,
            "payment_electronic_check": 0
        }
    }
]

# -------------------------------------------------------------------------
# RUN PREDICTIONS
# -------------------------------------------------------------------------
for case in test_cases:
    print("-" * 70)
    print(f"Customer: {case['name']}")
    print(f"Profile: {case['description']}")

    result = predictor.predict(case["data"])

    prob = result["churn_probability"]

    print(f"\n📈 Churn Probability: {prob:.2%}")
    print(f"🔮 Prediction: {result['prediction']}")
    print(f"⚠️  Risk Level: {result['risk_level'].upper()}")

    if prob >= 0.7:
        print("🔴 HIGH RISK — Immediate retention action")
    elif prob >= 0.4:
        print("🟡 MEDIUM RISK — Monitor closely")
    else:
        print("🟢 LOW RISK — Stable customer")

    print()

print("=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
