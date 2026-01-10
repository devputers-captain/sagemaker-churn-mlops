"""
============================================================================
CLEANUP SAGEMAKER RESOURCES
============================================================================
This deletes all old endpoints, configs, and models to start fresh
============================================================================
"""

import time

import boto3

REGION = "ap-south-1"
ENDPOINT_NAME = "churn-prediction-endpoint-v2" # CHANGE THIS

sm = boto3.client("sagemaker", region_name=REGION)

print("="*70)
print("CLEANING UP SAGEMAKER RESOURCES")
print("="*70)

# ============================================================================
# STEP 1: Delete Endpoints
# ============================================================================
print("\n[1/3] Checking for endpoints...")

try:
    endpoints = sm.list_endpoints(
        NameContains='churn',
        MaxResults=50
    )
    
    if endpoints['Endpoints']:
        print(f"Found {len(endpoints['Endpoints'])} endpoint(s):")
        
        for endpoint in endpoints['Endpoints']:
            name = endpoint['EndpointName']
            status = endpoint['EndpointStatus']
            print(f"  - {name} ({status})")
            
            if status in ['InService', 'Failed', 'OutOfService']:
                try:
                    print(f"    Deleting...")
                    sm.delete_endpoint(EndpointName=name)
                    print(f"    ✓ Deleted")
                except Exception as e:
                    print(f"    ⚠ Error: {e}")
        
        print("\n  Waiting 10 seconds for endpoints to delete...")
        time.sleep(10)
    else:
        print("  No endpoints found")
        
except Exception as e:
    print(f"  Error listing endpoints: {e}")

# ============================================================================
# STEP 2: Delete Endpoint Configurations
# ============================================================================
print("\n[2/3] Checking for endpoint configurations...")

try:
    configs = sm.list_endpoint_configs(
        NameContains='churn',
        MaxResults=50
    )
    
    if configs['EndpointConfigs']:
        print(f"Found {len(configs['EndpointConfigs'])} config(s):")
        
        for config in configs['EndpointConfigs']:
            name = config['EndpointConfigName']
            print(f"  - {name}")
            
            try:
                sm.delete_endpoint_config(EndpointConfigName=name)
                print(f"    ✓ Deleted")
            except Exception as e:
                print(f"    ⚠ Error: {e}")
    else:
        print("  No endpoint configs found")
        
except Exception as e:
    print(f"  Error listing configs: {e}")

# ============================================================================
# STEP 3: Delete Models
# ============================================================================
print("\n[3/3] Checking for models...")

try:
    models = sm.list_models(
        NameContains='churn',
        MaxResults=50
    )
    
    if models['Models']:
        print(f"Found {len(models['Models'])} model(s):")
        
        for model in models['Models']:
            name = model['ModelName']
            print(f"  - {name}")
            
            try:
                sm.delete_model(ModelName=name)
                print(f"    ✓ Deleted")
            except Exception as e:
                print(f"    ⚠ Error: {e}")
    else:
        print("  No models found")
        
except Exception as e:
    print(f"  Error listing models: {e}")

print("\n" + "="*70)
print("CLEANUP COMPLETE!")
print("="*70)

print("""
All old resources have been deleted.

What was NOT deleted (safe to keep):
------------------------------------
✓ Model Registry entries (ChurnModelPackageGroup)
✓ Pipeline definitions
✓ S3 data and model artifacts
✓ Training jobs history

Now you can redeploy cleanly:
----------------------------
python deploy-with-inference.py

Or start over completely:
------------------------
python pipeline.py              # Retrain model
python deploy-with-inference.py # Deploy it
python test-predictions.py      # Test it
""")

print("="*70)