# chatgpt_diy_qa_robot
一个通用的基于chatgpt api的问答机器人。你的数据集投喂给它什么，他就能智能的回答相关领域的问题，成为你的专属百问百答小助理。

常见问题：
报错Retrying langchain.embeddings.openai.embed_with_retry.<locals>._embed_with_retry in 4.0 seconds as it raised APIConnectionError: Error communicating with OpenAI: HTTPSConnectionPool(host='api.openai.com', port=443): Max retries exceeded with url: /v1/engines/text-embedding-ada-002/embeddings (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x0000022E4FC167F0>: Failed to establish a new connection: [WinError 10060] ：
原因定位了SSL处，具体不详。参见：https://www.zhihu.com/question/587322263/answer/2919916984来解决


报错openai.error.AuthenticationError: <empty message>：
是因为openai的key可能被官方干掉了，官网说：为了保护您帐户的安全，OpenAI 还可能会自动轮换我们发现已公开泄露的任何 API 密钥

