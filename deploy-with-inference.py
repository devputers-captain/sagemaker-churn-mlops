"""
============================================================================
DEPLOY MODEL WITH INFERENCE SCRIPT
============================================================================
This creates a new model that includes the inference.py script,
then deploys it to an endpoint.
============================================================================
"""

import os
import time

import boto3
from sagemaker import Session, get_execution_role
from sagemaker.sklearn import SKLearnModel

# Configuration
REGION = "ap-south-1"
MODEL_PACKAGE_GROUP = "ChurnModelPackageGroup"
ENDPOINT_NAME = "churn-prediction-endpoint-v2"
BUCKET = "sagemaker-ap-south-1-542810186858"
ROLE = get_execution_role()
sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
sagemaker_session = Session()

print("="*70)
print("DEPLOYING MODEL WITH INFERENCE SCRIPT")
print("="*70)

# Step 1: Get the latest trained model artifact
print("\n[1/6] Getting latest model artifact...")

models = sm.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)

if not models['ModelPackageSummaryList']:
    print("❌ No models found! Run your pipeline first.")
    exit(1)

model_arn = models['ModelPackageSummaryList'][0]['ModelPackageArn']
model_package = sm.describe_model_package(ModelPackageName=model_arn)

# Get the original model artifact S3 path
original_model_data = model_package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
print(f"✓ Original model: {original_model_data}")

# Step 2: Check if inference.py exists
print("\n[2/6] Checking for inference.py...")

if not os.path.exists('src/inference.py'):
    print("❌ inference.py not found!")
    print("\nPlease create inference.py first:")
    print("  1. Copy the inference.py script from the artifacts")
    print("  2. Save it in the same directory as this script")
    print("  3. Run this script again")
    exit(1)

print("✓ Found inference.py")

# Step 3: Create model package with inference script
print("\n[3/6] Creating model with inference script...")

# Create SKLearnModel with inference script
model = SKLearnModel(
    model_data=original_model_data,
    role=ROLE,
    entry_point='src/inference.py',
    framework_version='1.2-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

print("✓ Model package created with inference script")

# Step 4: Delete old endpoint if exists
print("\n[4/6] Checking for existing endpoint...")

try:
    existing = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"⚠ Endpoint exists with status: {existing['EndpointStatus']}")
    print("  Deleting old endpoint...")

    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print("  Waiting for deletion...")
    time.sleep(30)  # Wait for deletion
    print("✓ Old endpoint deleted")

except sm.exceptions.ClientError:
    print("✓ No existing endpoint to delete")

# Step 5: Deploy the model
print("\n[5/6] Deploying model to endpoint...")
print("⏳ This takes 5-10 minutes...")

endpoint_config_name = f"{ENDPOINT_NAME}-{int(time.time())}"

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name=ENDPOINT_NAME,
    endpoint_config_name=endpoint_config_name
)

print("✓ Deployment complete!")

# Step 6: Test the endpoint
print("\n[6/6] Testing endpoint using...")
print("\n python tests/test-predictions.py")

print("\n" + "="*70)
print("✅ SUCCESS!")
print("="*70)
