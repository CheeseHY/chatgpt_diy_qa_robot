#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：Cheese_AI
@File ：webui.py
@Author ：Cheese
@Date ：2023/6/13 18:16
'''
from langchain.document_loaders import UnstructuredWordDocumentLoader # 粘合剂:帮助粘合下面的需要用到的各种工具 # pip install langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import pinecone # 向量数据库 # pip install pinecone-client

import os
from env import OPENAI_API_KEY,PINECONE_API_KEY,PINECONE_API_ENV,INDEX_NAME


from langchain.vectorstores import Pinecone

import streamlit as st # 快速创建网站的开源工具 # pip install streamlit
import gtts # 谷歌文字转语音 # pip install gtts
from langchain.llms import OpenAI # 大语言模型帮助我们分析文档以及回答问题

from langchain.chains.question_answering import load_qa_chain
import pinecone # 向量数据库 # pip install pinecone-client
from env import OPENAI_API_KEY,PINECONE_API_KEY,PINECONE_API_ENV,INDEX_NAME
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os

import os
os.environ["http_proxy"]="127.0.0.1:4780" # 你电脑 -> 网络和Internet -> 代理 -> 使用代理服务器处 -> 取自这里的ip和port
os.environ["https_proxy"]="127.0.0.1:4780"
'''
存储完成向量数据库之后，我们就可以运行下面的代码，用streamlit帮我们做一个简单的网页可以用来调用我们的机器人问答
'''
# App framework
# 如何创建自己的网页机器人
st.title('😬交规小秘书😬') #用streamlit app创建一个标题
# 创建一个输入栏可以让用户去输入问题
query = st.text_input('嗨，我是你的私人交规小秘书,你可以问我关于交通法律的问题，例如：不按交通信号灯通行扣几分？')

my_bar = st.progress(0, text='等待投喂问题哦') # 美化：加载进度条
# initialize search
# 开始搜索，解答
if query:
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    #llm是用来定义大型语言模型，在下面的例子，用的是openai，注意，此openai调用的是langchain方法不是openai本ai
    llm = OpenAI(temperature=0, max_tokens=-1, openai_api_key=OPENAI_API_KEY) # temperature=0给我非常确认的答案，不需要创意
    print('1:'+ str(llm))
    my_bar.progress(10, text='正在查询新华字典')
    # embedding就是把文字变成数字
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print('2:'+ str(embeddings))
    # 调用pinecone的数据库，开始查询任务
    docsearch = Pinecone.from_existing_index('cheese0',embedding=embeddings)
    print('3:'+ str(docsearch))
    # 相似度搜索，例如疼678，痛679，搜索用户的问题的相似度
    docs = docsearch.similarity_search(query, k=3) # 匹配到最相关的是k个文档给我，每个文档上边设置了是500个字 # include_metadata=True # openai原理：embedding向量化或semantic search语义搜索。采取高位相似度的近似度的查询。假设数据集中有"扣"分，你问他“减”分，虽然数据集中没有，但是扣在向量库中是788，减在openai中是789，openai的模型中已经预设了，知道他们是近义词。
    print('4:'+ str(docs))
    my_bar.progress(60, text='找到点头绪了')
    # 调用langchain的load qa办法，’stuff‘为一种放入openai的办法
    chain = load_qa_chain(llm, chain_type='stuff', verbose=True) # stuff 硬塞进去
    print('5:'+ str(chain))
    my_bar.progress(90, text='可以开始生成答案了，脑细胞在燃烧')
    # 得到答案
    answer = chain.run(input_documents=docs, question=query, verbose=True)
    print('6:'+ str(answer))

    my_bar.progress(100, text='好了') # 美化的东西
    st.write(answer)
    audio = gtts.gTTS(answer, lang='zh') # 文字转语音
    audio.save("audio.wav")
    st.audio('audio.wav', start_time=0)
    os.remove("audio.wav")