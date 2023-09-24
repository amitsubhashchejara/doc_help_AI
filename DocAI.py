# IMPORTING ALL THE NEEDED MODULES AND LIBRARIES

# all the telegram bot modules
import asyncio
import telegram
from telegram import Update, InputFile, _message
import os
from telegram.ext import Updater, filters, CommandHandler, CallbackContext, ConversationHandler, ApplicationBuilder, ContextTypes, MessageHandler, filters, ApplicationBuilder, CallbackQueryHandler
import tracemalloc



# all the openai modules
import openai


# all the langchain modules
import langchain
from langchain.document_loaders import TelegramChatFileLoader, TelegramChatApiLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

tracemalloc.start()

#GETTING ALL THE ESSENTIAL API'S 

from dotenv import load_dotenv, find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key=os.getenv('api_key_openai_1')
bot = telegram.Bot(os.getenv('T_bot_api_2'))
TOKEN=os.getenv('T_bot_api_2')

llm_name="gpt-3.5-turbo"
llm=ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=os.getenv('api_key_openai_1'))

BOT_ID=6596826073
application = ApplicationBuilder().token(TOKEN).build()

#DEFINING DIFFERENT FUNCTIONS FOR EVERY LIBRARY

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response=openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    # print(str(response.choices[0].message))
    return response.choices[0].message["content"]

# TO SEND THE FIRST MESSAGE TO THE USER

async def start(update: Update, context: CallbackContext):
    user = update.effective_user  # Get the user object
    user_id = user.id
    if 'greeted' not in context.user_data:
        # This user has not been greeted yet
        user_input=update.message.text
        context.user_data['greeted'] = True
        sticker_id = "CAACAgIAAxkBAAEf2kJkPWxJeC0zSz1kOizkm-BcpKQO5gACAQEAAladvQoivp8OuMLmNC8E"
        await update.message.reply_sticker(sticker_id)
        await update.message.reply_text('HELLO !!!\nWelcome to DocHelpAI')
        await update.message.reply_text('Please click on manu button and select \'doc_help\' to\nget help related to document or just chat with me\nin general here.')

async def doc_help(update: Update, context: CallbackContext):
    await update.message.reply_text('Send me a document file to get help.')
    user_input_doc=update.message.document

    file_id = user_input_doc.file_id
    file_name = user_input_doc.file_name

    file = context.bot.get_file(file_id)

    
    
        
    loader = TelegramChatFileLoader(user_input_doc)
    pages=loader.load()

    #splitting the document
    r_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300, 
    separators=["\n\n", "\n", " ", ""]
    )
    splitted_pages=r_splitter.split(pages)
    
    #embedding the document
    embedding=OpenAIEmbeddings(openai_api_key=openai.api_key)
    global db
    db=DocArrayInMemorySearch.from_documents(
    splitted_pages, 
    embedding=embedding,
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Document processed successfully!!!\nYou can ask me questions related to this document now.")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Ask me questions related to this document")
    user_input_4=update.message.text
        
    # Build prompt
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT=PromptTemplate.from_template(template)  

    qa_chain=RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    question=user_input_4
    result=qa_chain({"query": question})
    context.bot.send_message(chat_id=update.effective_chat.id, text=result)





        


       


    




# Handle Document input 

# def doc_help(update: Update, context: CallbackContext):
#     user = update.effective_user  # Get the user object
#     user_id = user.id
#     #loading the document
#     user_input = update.message
#     context.bot.send_message(chat_id=update.effective_chat.id, text="hi you are in doc help function") 
#     if user_input.document:
#         context.bot.send_message(chat_id=update.effective_chat.id, text="Processing the document...")
#         loader = TelegramChatFileLoader(user_input)
#         pages=loader.load()

#         #splitting the document
#         r_splitter=RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=300, 
#         separators=["\n\n", "\n", " ", ""]
#         )
#         splitted_pages=r_splitter.split(pages)
    
#         #embedding the document
#         embedding=OpenAIEmbeddings(openai_api_key=openai.api_key)
#         global db
#         db=DocArrayInMemorySearch.from_documents(
#         splitted_pages, 
#         embedding=embedding,
#         )
#         context.bot.send_message(chat_id=update.effective_chat.id, text="Document processed successfully!!!\nYou can ask me questions related to this document now.")

#         user_input=update.message
#         if user_input.text: 
#             # Build prompt
#             template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
#             {context}
#             Question: {question}
#             Helpful Answer:"""
#             QA_CHAIN_PROMPT=PromptTemplate.from_template(template)  

#             qa_chain=RetrievalQA.from_chain_type(
#             llm,
#             retriever=db.as_retriever(),
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#             )

#             question=update.message.text 
#             result=qa_chain({"query": question})
#             context.bot.send_message(chat_id=update.effective_chat.id, text=result)

application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("doc_help",doc_help))

if __name__ == '__main__':
    # application = ApplicationBuilder().token(TOKEN).build()
    # application.add_handler(CommandHandler("start", start))
    # application.add_handler(CommandHandler("doc_input", doc_input))
    # document_handler = MessageHandler(filters.document, doc_input)

    # echo_handler_1 = MessageHandler(filters.TEXT & (~filters.COMMAND), start)
    # application.add_handler(echo_handler_1)
    application.run_polling()
    



