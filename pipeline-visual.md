# SageMaker Pipeline Architecture - Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TRIGGER: EventBridge Schedule (Lambda Trigger)                          │
│ OR: Manual execution via Python/CLI/Console                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         📊 INPUT DATA                                    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ S3: s3://bucket/churn-pipeline/input-data/customer_data.csv    │    │
│  │                                                                  │    │
│  │ Contains:                                                        │    │
│  │ - CustomerID, Gender, Age                                        │    │
│  │ - MonthlyCharges, TotalCharges, Tenure                          │    │
│  │ - Contract, InternetService, TechSupport                        │    │
│  │ - StreamingTV, PaymentMethod                                    │    │
│  │ - Churn (Target: Yes/No)                                        │    │
│  └────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
╔═════════════════════════════════════════════════════════════════════════╗
║  STEP 1: DATA PREPROCESSING                                             ║
║  Script: preprocessing.py                                               ║
║  Instance: ml.m5.xlarge                                                 ║
║  Duration: ~2-3 minutes                                                 ║
╚═════════════════════════════════════════════════════════════════════════╝
│
│  What happens inside:
│  1. Load CSV from S3
│  2. Handle missing values (median for numbers, mode for categories)
│  3. Remove duplicates
│  4. Encode categorical variables (Male/Female → 0/1)
│  5. Normalize numerical features (scale to similar ranges)
│  6. Split data:
│     - 70% Training   (learn patterns)
│     - 20% Test       (final evaluation)
│     - 10% Validation (tune model)
│
│  Output to S3:
│  ├── train.csv       → Used for training
│  ├── validation.csv  → Used for validation during training
│  └── test.csv        → Used for final evaluation
│
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
╔═════════════════════════════════════════════════════════════════════════╗
║  STEP 2: MODEL TRAINING                                                 ║
║  Script: training.py                                                    ║
║  Instance: ml.m5.xlarge                                                 ║
║  Duration: ~5-10 minutes                                                ║
╚═════════════════════════════════════════════════════════════════════════╝
│
│  What happens inside:
│  1. Load train.csv and validation.csv
│  2. Train Random Forest Classifier with:
│     - 100 decision trees
│     - Max depth of 10 levels
│     - Min 4 samples to split a node
│  3. Validate performance during training
│  4. Package trained model
│
│  Model learns:
│  "Customers with month-to-month contracts, high monthly charges,
│   no tech support, and short tenure are likely to churn"
│
│  Output to S3:
│  └── model.tar.gz  (Contains trained model.joblib)
│
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
╔═════════════════════════════════════════════════════════════════════════╗
║  STEP 3: MODEL EVALUATION                                               ║
║  Script: evaluation.py                                                  ║
║  Instance: ml.m5.xlarge                                                 ║
║  Duration: ~1-2 minutes                                                 ║
╚═════════════════════════════════════════════════════════════════════════╝
│
│  What happens inside:
│  1. Load trained model and test.csv
│  2. Make predictions on test data (never seen before)
│  3. Calculate performance metrics:
│     
│     ┌─────────────────────────────────────────┐
│     │ Accuracy:  How often is model correct? │
│     │ Precision: Of predicted churns, % real  │
│     │ Recall:    Of real churns, % caught    │
│     │ F1 Score:  Balanced performance measure │
│     └─────────────────────────────────────────┘
│
│  Example output:
│  {
│    "metrics": {
│      "accuracy": 0.82,    ← 82% correct predictions
│      "precision": 0.78,   ← 78% of churn predictions are right
│      "recall": 0.75,      ← Catches 75% of actual churns
│      "f1_score": 0.76
│    },
│    "confusion_matrix": {
│      "true_negatives": 65,   ← Correctly predicted: won't churn
│      "false_positives": 5,   ← Incorrectly predicted: will churn
│      "false_negatives": 8,   ← Missed: actually churned
│      "true_positives": 22    ← Correctly predicted: will churn
│    }
│  }
│
│  Output to S3:
│  └── evaluation.json  (Contains all metrics)
│
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
╔═════════════════════════════════════════════════════════════════════════╗
║  STEP 4: CONDITIONAL REGISTRATION                                       ║
║  Condition: IF accuracy >= 0.75 (75%)                                   ║
╚═════════════════════════════════════════════════════════════════════════╝
│
│  Decision Logic:
│  ┌──────────────────────────────────────────────────────┐
│  │  IF model accuracy >= 75%                            │
│  │    THEN: Register model in Model Registry            │
│  │          Status: PendingManualApproval               │
│  │          → Ready for production deployment           │
│  │                                                       │
│  │  ELSE: Do NOT register model                         │
│  │        Pipeline ends here                            │
│  │        Team notified of poor performance             │
│  └──────────────────────────────────────────────────────┘
│
│  If registered:
│  ├── Model Package created in SageMaker
│  ├── Version number assigned (v1, v2, v3...)
│  ├── All metadata attached (accuracy, timestamp, data source)
│  └── Available for deployment to production
│
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ✅ PIPELINE COMPLETE                             │
│                                                                           │
│  Total Duration: ~10-15 minutes                                          │
│  Cost per run: ~$0.06                                                    │
│  Next run: Tomorrow at 2 AM                                              │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
                          MONITORING & ACCESS
═══════════════════════════════════════════════════════════════════════════

📊 View in AWS Console:
   SageMaker → Pipelines → CustomerChurnPredictionPipeline
   
   Graph View Shows:
   [Preprocessing] → [Training] → [Evaluation] → [Condition] → [Register]
        ✓               ✓            ✓             ✓             ✓
   
📈 Access Results:
   - Training metrics: S3 → models/
   - Evaluation report: S3 → output-data/evaluation/evaluation.json
   - Model artifacts: S3 → models/model.tar.gz
   

═══════════════════════════════════════════════════════════════════════════
                        REAL BUSINESS IMPACT
═══════════════════════════════════════════════════════════════════════════

📉 Without Pipeline:
   - Manual process takes 4+ hours per week
   - Data scientist runs scripts manually
   - Models not updated regularly
   - Stale predictions used for months
   - Miss early warning signs
   - Higher churn rate
   
📈 With Pipeline:
   - Fully automated, runs daily while you sleep
   - Always using fresh data
   - Models retrained weekly
   - Predictions updated daily
   - Catch at-risk customers early
   - 20-30% reduction in churn rate

✅ What Control:
   - When pipeline runs (scheduling)
   - What instance types to use (cost optimization)
   - Where data comes from (S3, databases, APIs)
   - How steps connect (dependencies)
   - Failure handling and retries
   - Monitoring and alerting
   - Cost management
   - Security and permissions
   