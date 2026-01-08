"""
============================================================================
CHECK ENDPOINT CLOUDWATCH LOGS
============================================================================
This retrieves the latest error logs from your endpoint
============================================================================
"""

import boto3
from datetime import datetime, timedelta

ENDPOINT_NAME = "churn-prediction-endpoint"
REGION = "ap-south-1"

logs = boto3.client('logs', region_name=REGION)

print("="*70)
print("CHECKING ENDPOINT LOGS")
print("="*70)

# CloudWatch log group for endpoints
log_group = f"/aws/sagemaker/Endpoints/{ENDPOINT_NAME}"

print(f"\nLog Group: {log_group}")
print("\nFetching latest logs (last 10 minutes)...\n")

try:
    # Get log streams (each instance has its own stream)
    streams = logs.describe_log_streams(
        logGroupName=log_group,
        orderBy='LastEventTime',
        descending=True,
        limit=5
    )
    
    if not streams['logStreams']:
        print("❌ No log streams found")
        print("   Endpoint might not have started yet")
        exit(1)
    
    print(f"✓ Found {len(streams['logStreams'])} log stream(s)\n")
    print("="*70)
    
    # Get logs from the most recent stream
    for stream in streams['logStreams'][:2]:  # Check top 2 streams
        stream_name = stream['logStreamName']
        print(f"\nStream: {stream_name}")
        print("-"*70)
        
        # Get recent log events
        start_time = int((datetime.now() - timedelta(minutes=10)).timestamp() * 1000)
        
        events = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            startTime=start_time,
            limit=100,
            startFromHead=False  # Get most recent
        )
        
        # Print log events
        error_found = False
        for event in events['events']:
            message = event['message']
            timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
            
            # Highlight errors
            if any(keyword in message.lower() for keyword in ['error', 'exception', 'traceback', 'failed']):
                print(f"\n🔴 [{timestamp}]")
                print(message)
                error_found = True
            elif 'warning' in message.lower():
                print(f"\n⚠️  [{timestamp}]")
                print(message)
        
        if not error_found:
            # Show all logs if no errors found
            print("\nAll recent logs:")
            for event in events['events'][-20:]:  # Last 20 logs
                message = event['message']
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                print(f"[{timestamp}] {message}")
    
    print("\n" + "="*70)
    print("COMMON ISSUES & SOLUTIONS")
    print("="*70)
    
    print("""
If you see errors about:

1. "No module named 'joblib'" or missing imports:
   → Model was trained with different dependencies
   → Solution: Update training.py to save model with all dependencies

2. "model.joblib not found" or "No such file":
   → Model artifact structure is wrong
   → Solution: Check model packaging in training.py

3. "AttributeError" or "NoneType":
   → Input data format doesn't match expected format
   → Solution: Check inference.py input parsing

4. "predict_proba" not found:
   → Model object doesn't have this method
   → Solution: Verify model is a sklearn classifier

Most Common Fix:
---------------
The model artifact might not be packaged correctly. Let's verify:
    python check-model-artifact.py
""")
    
except logs.exceptions.ResourceNotFoundException:
    print(f"❌ Log group not found: {log_group}")
    print("\nThis means the endpoint hasn't started yet or failed to start.")
    print("\nCheck endpoint status:")
    print(f"  aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME}")
    
except Exception as e:
    print(f"❌ Error reading logs: {e}")
    print("\nTry viewing logs in AWS Console:")
    print(f"  https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#logEventViewer:group=/aws/sagemaker/Endpoints/{ENDPOINT_NAME}")

print("\n" + "="*70)