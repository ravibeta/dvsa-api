from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://found-vision-1.services.ai.azure.com/openai/v1"
deployment_name = "text-embedding-ada-002"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)

response = client.responses.create(
    model=deployment_name,
    input="What is the capital of France?",
)

print(f"answer: {response.output[0]}")
