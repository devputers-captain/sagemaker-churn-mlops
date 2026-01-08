# 🚀 End-to-End Churn Prediction MLOps Pipeline on AWS SageMaker

This project demonstrates a **production-grade MLOps workflow** using **AWS SageMaker**, focused on training, validating, registering, and deploying a customer churn prediction model.

The goal is to show how **existing ML code** can be **operationalized** using SageMaker Pipelines, Model Registry, and real-time inference endpoints.

---

## 🧠 Problem Statement

Customer churn is a critical business problem.  
This project builds an automated ML pipeline that:

- Preprocesses customer data
- Trains a churn prediction model
- Evaluates model quality
- Prevents low-quality models from deploying
- Deploys approved models to a real-time endpoint
- Supports monitoring and retraining

---

## 🏗️ Architecture Overview

```

S3 (Input Data)
↓
ProcessingStep (Preprocessing)
↓
TrainingStep (Model Training)
↓
ProcessingStep (Evaluation)
↓
ConditionStep (Quality Gate)
↓
Model Registry
↓
Approved Model
↓
Real-Time Endpoint

```

---

## 🛠️ Tech Stack

- **Python**
- **AWS SageMaker**
  - SageMaker Pipelines
  - Training Jobs
  - Processing Jobs
  - Model Registry
  - Real-time Endpoints
- **scikit-learn**
- **Amazon S3**
- **Amazon CloudWatch**
- **GitHub**

---

## 📁 Repository Structure

```

sagemaker-churn-mlops/
├── src/
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
│   └── inference.py       # Inference interface
├── pipeline.py/ipynd      # SageMaker Pipeline definition
├── requirements.txt
└── README.md

````

---

## ⚙️ Pipeline Steps Explained

### 1️⃣ Preprocessing
- Cleans and prepares raw data
- Outputs processed training data

### 2️⃣ Training
- Trains a scikit-learn churn prediction model
- Saves model artifact to S3

### 3️⃣ Evaluation
- Evaluates model performance (e.g., accuracy)
- Writes metrics to `evaluation.json`

### 4️⃣ Quality Gate
- Uses a `ConditionStep`
- Only allows models with acceptable metrics to proceed

### 5️⃣ Model Registration
- Registers approved models in SageMaker Model Registry
- Enables versioning and governance

---

## 🚀 How to Run the Pipeline

### Prerequisites
- AWS account
- SageMaker Notebook or Studio
- IAM role with SageMaker permissions

---

### Step 1️⃣ Clone the Repository in SageMaker Notebook

```bash
git clone https://github.com/devputers-captain/sagemaker-churn-mlops.git
cd sagemaker-churn-mlops
````

---

### Step 2️⃣ Install Dependencies

```bash
!pip install -r requirements.txt
```

---

### Step 3️⃣ Run the Pipeline

```python
from pipeline import pipeline
import sagemaker

role = sagemaker.get_execution_role()

pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

---

### Step 4️⃣ Monitor Execution

In AWS Console:

```
SageMaker → Pipelines → ChurnPipeline
```

---

## 🔮 Inference

The deployed model exposes a **real-time endpoint** that accepts JSON input:

```json
{
  "gender_male": 1,
  "age": 45,
  "monthly_charges": 29.85,
  "total_charges": 1500.5,
  "tenure": 50,
  "contract_month_to_month": 0,
  "contract_two_year": 1,
  "internet_dsl": 1,
  "tech_support_yes": 1,
  "streaming_tv_yes": 0,
  "payment_electronic_check": 0
}
```

Response:

```json
{
  "churn_probability": 0.03,
  "prediction": "no_churn",
  "risk_level": "low"
}
```
---

## 📌 Future Enhancements

* Add SageMaker Model Monitor for drift detection
* Automate retraining
* CI/CD integration using GitHub Actions or CodePipeline
* Add SHAP-based model explainability

