#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：Cheese_AI
@File ：main.py
@Author ：Cheese
@Date ：2023/6/13 16:25
'''
# pip install unstructured
# pip install openai
# pip install tiktoken

from langchain.document_loaders import UnstructuredWordDocumentLoader # 粘合剂:帮助粘合下面的需要用到的各种工具 # pip install langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import pinecone # 向量数据库 # pip install pinecone-client

import os
os.environ["http_proxy"]="127.0.0.1:4780" # 你电脑 -> 网络和Internet -> 代理 -> 使用代理服务器处 -> 取自这里的ip和port
os.environ["https_proxy"]="127.0.0.1:4780"
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

'''
下面的这部分代码是将文件夹中的word文档，上传到自己的向量数据库
'''
#首先进入文件夹查看数据
directory_path = 'data' #这边填入你自己的数据文件所在的文件夹
data = []
# loop through each file in the directory
for filename in os.listdir(directory_path):
    # check if the file is a doc or docx file
    # 检查所有doc以及docx后缀的文件
    if filename.endswith(".doc") or filename.endswith(".docx"):
        # print the file name
        # langchain自带功能，加载word文档
        loader = UnstructuredWordDocumentLoader(f'{directory_path}/{filename}')
        print(loader)
        data.append(loader.load())
print(len(data))
#Chunking the data into smaller pieces
#再用菜刀把文档分隔开，chunk_size就是我们要切多大，建议设置700及以下，因为openai有字数限制，chunk_overlap就是重复上下文多少个字
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = []
for i in range(len(data)):
    print(i)
    texts.append(text_splitter.split_documents(data[i]))
    print(text_splitter.split_documents(data[i]))
print(len(texts))

#Creating embeddings
# 把文字转成数字
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
# 把数字放进向量数据库，environment填写你的数据库所在的位置，例如useast
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
# 要填入对应的index name
index_name = INDEX_NAME # put in the name of your pinecone index here
for i in range(len(texts)):
    Pinecone.from_texts([t.page_content for t in texts[i]], embeddings, index_name=index_name)
    print("done")

# ------------------



# '''
# 以下代码删除所有indexes模块，默认注释关闭，谨慎使用
# '''
# Are you sure you want to delete this? If yes, delete this line of code
# # deleting all indexes
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_API_ENV
# )
# pinecone.Index('dental').delete(delete_all=True)
