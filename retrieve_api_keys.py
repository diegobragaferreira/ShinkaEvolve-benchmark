import os
import sys
import boto3

def main():
    api_base = "https://litellm.genai.local"
    ssm_param = os.getenv("SSM_PARAM", "/MIND/PRD/EVOLVE")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

    try:
        ssm = boto3.client("ssm", region_name=region)
        response = ssm.get_parameter(Name=ssm_param, WithDecryption=True)
        api_key = response["Parameter"]["Value"]
        
        # Output commands for the shell to evaluate
        print(f"export API_BASE='{api_base}'")
        print(f"export API_KEY='{api_key}'")
        
    except Exception as e:
        print(f"Error retrieving parameters from SSM: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()