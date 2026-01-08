import urllib.parse

import boto3

sagemaker = boto3.client("sagemaker")

PIPELINE_NAME = "CustomerChurnPredictionPipelineV2"  # CHANGE THIS

def lambda_handler(event, context):
    print("Event received:", event)

    # S3 direct trigger format
    record = event["Records"][0]

    bucket = record["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

    print(f"New file uploaded: s3://{bucket}/{key}")

    # Safety check
    if not key.endswith(".csv"):
        print("Not a CSV file, skipping")
        return

    input_data_url = f"s3://{bucket}/{key}"

    response = sagemaker.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineParameters=[
            {
                "Name": "InputDataUrl",
                "Value": input_data_url
            }
        ]
    )

    print("Pipeline started:")
    print(response["PipelineExecutionArn"])