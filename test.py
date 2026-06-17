import asyncio
from code_switch.llm import LocalClient, Message
c = LocalClient()  # or LocalClient(model="Qwen/Qwen2.5-0.5B-Instruct") for a fast check
print(asyncio.run(c.complete([Message(role="user", content="Say hi in English and Spanish.")], max_new_tokens=64)).text)
