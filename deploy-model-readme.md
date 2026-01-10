# Customer Churn Prediction - Deployment Guide

## 🚀 Quick Start

### Deploy Your Model
```bash
python deploy-with-inference.py
```

**Expected time:** 5-10 minutes  

### Test Predictions
```bash
python tests/test-predictions.py
```

---

## 📋 Prerequisites

Before deploying, ensure you have:

- [x] Trained model in Model Registry (`ChurnModelPackageGroup`)
- [x] `inference.py` file in src directory
- [x] AWS credentials configured
- [x] Required IAM permissions (SageMaker, S3)

---

## 🔧 Common Issues & Fixes

### ❌ Error: "endpoint configuration already exists"

**Cause:** Old endpoint config wasn't deleted

**Fix:**
```bash
python src/cleanup-sagemaker.py  # Delete all old resources
python deploy-with-inference.py  # Deploy fresh
```

---

### ❌ Error: "inference.py not found"

**Cause:** Missing inference script

**Fix:**
```bash
# Create inference.py in src directory with the provided template
# Then run deployment again
python deploy-with-inference.py
```

---

### ❌ Error: "Could not access model data at s3://..."

**Cause:** IAM role lacks S3 permissions

**Fix:**
```bash
# Add S3 permissions to your role
aws iam attach-role-policy \
    --role-name AmazonSageMaker-ExecutionRole-20251230T121523 \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

---

### ❌ Error: "500 Internal Server Error" when making predictions

**Cause:** Inference code failing inside endpoint

**Fix:**
```bash
# Check logs to see exact error
python check-endpoint-logs.py
```

---

### ❌ Error: "Endpoint not found" when testing

**Cause:** Endpoint wasn't created or is still deploying

**Fix:**
```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name churn-prediction-endpoint

# If status is "Creating", wait 5-10 minutes
# If status is "Failed", check logs:
python check-endpoint-logs.py
```
---

## 🔄 Complete Workflow

```bash
# 1. Train model (if not already trained)
run pipeline.ipynb

# 2. Clean up old resources (if required)
python cleanup-sagemaker.py

# 3. Deploy model
python deploy-with-inference.py

# 4. Test predictions
python test-predictions.py

# 5. Use in production
# See code examples in test-predictions.py
```

---

## 💰 Cost Management

**Active endpoint costs:** ~$0.23/hour = ~$165/month (Approx)

**To stop charges:**
```bash
aws sagemaker delete-endpoint --endpoint-name churn-prediction-endpoint
```

**To reduce costs:**
- Delete endpoint when not in use
- Use Batch Transform for bulk predictions (pay per use)

---

## 🎯 Quick Troubleshooting Checklist

- [ ] Ran `cleanup-sagemaker.py` to clear old resources?
- [ ] `inference.py` exists in current directory?
- [ ] IAM role has S3 and SageMaker permissions?
- [ ] Model exists in Model Registry?
- [ ] Waited 5-10 minutes for deployment?
- [ ] Checked logs with `check-endpoint-logs.py`?

If all checked and still failing, run:
```bash
python check-endpoint-logs.py
```
And share the error output.

---

## 📞 Support

**Check deployment status:**
```bash
aws sagemaker describe-endpoint --endpoint-name churn-prediction-endpoint
```

**View in AWS Console:**
```
https://ap-south-1.console.aws.amazon.com/sagemaker/home?region=ap-south-1#/endpoints
```

**CloudWatch Logs:**
```
https://ap-south-1.console.aws.amazon.com/cloudwatch/home?region=ap-south-1#logEventViewer:group=/aws/sagemaker/Endpoints/churn-prediction-endpoint
```

---

## ✅ Success Indicators

Deployment succeeded when you see:

```
✅ SUCCESS!
Endpoint Details:
----------------
Name: churn-prediction-endpoint
Status: InService

Now you can make predictions:
python test-predictions.py
```

And test shows:
```
Customer: Alice - High Risk Customer
📈 Churn Probability: 85.00%
🔴 HIGH RISK - Will Likely Churn
```
