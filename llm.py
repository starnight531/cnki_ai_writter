import openai

class LLM:
    def __init__(self, base_url = None, api_key = None):
        if not base_url:
            base_url = "http://127.0.0.1:10416/v1"
        if not api_key:
            api_key = "any ok"
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
 
    def generate_response(self, user_input, model):
        response = self.client.chat.completions.create(
            messages=[  
                {"role": "system", "content": ''},
                {"role": "user", "content": user_input}
            ],
            model=model,
        )
        response = response.choices[0].message.content
        return response

if __name__ == "__main__":
    user_input = "你是谁"
    models= ["R1-distill-Qwen2.5-1.5B", "R1-distill-Qwen2.5-7B_GGUF", "Qwen2.5-3B-Instruct", "QwQ-32B"]

    # llm = LLM()
    # res = llm.generate_response(user_input,models[1]).split("</think>")

    llm = LLM("http://10.25.160.22:10425/v1")
    res = llm.generate_response(user_input,models[3]).split("</think>")
    
    if len(res[0].strip()) == 0:
        print(res[1].strip())
    else:
        print(res[0].strip())
