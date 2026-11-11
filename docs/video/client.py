from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct", stream=True, max_tokens=60, temperature=0,
    messages=[{"role": "user", "content": "Explain KV caching in two sentences."}])
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
