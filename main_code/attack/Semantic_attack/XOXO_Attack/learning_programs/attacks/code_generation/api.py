import json
import os

from anthropic import AnthropicVertex
from google.oauth2 import service_account
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


def convert_logprob(logprobs):
    return [{'token': logprob.token,'bytes':logprob.bytes, 'logprob':logprob.logprob} for logprob in logprobs]


def getenv(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set.")
    return value


class OpenAIConnector:
    client: OpenAI
    model: str

    def __init__(self, model: str, api: str) -> None:
        self.model = model
        self.client = self._get_client(api)

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(**kwargs):
            response = self.client.chat.completions.create(**kwargs)
            return self._serialize_completion(response)

        self.completion_with_backoff = completion_with_backoff

    def _get_client(self, api: str):
        if api == 'openai':
            return OpenAI(api_key=getenv("OPENAI_API_KEY"))
        if api == "openrouter":
            return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=getenv("OPENROUTER_API_KEY"))
        raise ValueError(f"Unknown API: {api}")

    def _serialize_completion(self, completion):
        return {
            "id": completion.id,
            "choices":
            [
                {
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                    "logprobs": convert_logprob(choice.logprobs.content),
                    "message":
                    {
                        "content": choice.message.content,
                        "role": choice.message.role,
                        "function_call":
                        {
                            "arguments": json.loads(choice.message.function_call.arguments) if choice.message.function_call and choice.message.function_call.arguments else None,
                            "name": choice.message.function_call.name
                        } if choice.message and choice.message.function_call else None
                    } if choice.message else None
                } for choice in completion.choices
            ],
            "created": completion.created,
            "model": completion.model,
            "object": completion.object,
            "system_fingerprint": completion.system_fingerprint,
            "usage": {
                "completion_tokens": completion.usage.completion_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        }

    def get_response(self, prompt: str):
        return self.completion_with_backoff(
            model=self.model,
            messages=prompt,
            temperature=0,
            logprobs=True,
            max_tokens=2048,
        )


class SonnetConnector:
    client: AnthropicVertex
    model: str

    def __init__(self, model: str) -> None:
        self.model = model

        gcp_cred_file = getenv("GCP_CRED_FILE")
        project_id = getenv("PROJECT_ID")
        model_location = getenv("MODEL_LOCATION")

        if not os.path.exists(gcp_cred_file):
            raise FileNotFoundError(f"API key file not found at {gcp_cred_file}")

        self.client = AnthropicVertex(
            region=model_location,
            project_id=project_id,
            credentials=service_account.Credentials.from_service_account_file(
                gcp_cred_file,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            ),
            timeout=10*60.,
        )

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def message_with_backoff(**kwargs):
            response = self.client.messages.create(**kwargs)
            return self._serialize_message(response)

        self.message_with_backoff = message_with_backoff

    def _serialize_message(self, message):
        return {
            "id": message.id,
            "choices":
            [
                {
                    "finish_reason": message.stop_reason,
                    "message": {"content": content.text}
                } for content in message.content
            ],
            "model": message.model,
            "object": message.type,
            "usage": {
                "completion_tokens": message.usage.output_tokens,
                "prompt_tokens": message.usage.input_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens
            }
        }

    def get_response(self, prompt):
        return self.message_with_backoff(
            model=self.model,
            messages=prompt,
            temperature=0.0,
            max_tokens=4096
        )
