# IMPORTING ALL THE NEEDED MODULES AND LIBRARIES

# all the telegram bot modules
import asyncio
import telegram
from telegram import Update
import os
from telegram.ext import Updater, filters, CommandHandler, CallbackContext, ConversationHandler, ApplicationBuilder, ContextTypes, MessageHandler, filters, ApplicationBuilder, CallbackQueryHandler
import tracemalloc
from telegram import CallbackQuery

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
        context.user_data['greeted'] = True
        user_input=update.message.text
        
        sticker_id = "CAACAgIAAxkBAAEf2kJkPWxJeC0zSz1kOizkm-BcpKQO5gACAQEAAladvQoivp8OuMLmNC8E"
        await update.message.reply_sticker(sticker_id)
        
        await update.message.reply_text('HELLO !!!\nWelcome to DocHelpAI')
        await update.message.reply_text('Send me a document file to start a Q&A session over it.')

        prompt="""You are a friendly Document Assistant.
        Your job is to help users find specific information from their documents or summarize them. 
        You will take the document file as an input and output the required content in a structured format. 
        You can communicate with the user, but strictly only regarding the files they have given as input and no other than that. 
        Please follow these instructions to fulfill your role:
        - Ask the user to upload their document and specify the type of document it is (PDF, Excel, Word, PPT, etc.)
        - Ask the user what information they are looking for or if they would like you to summarize the document for them
        - Provide the user with the information they need in a structured format such as JSON or HTML
        - Check whether all conditions are satisfied before providing the output
        - Only assist the user with the documents that they have provided as input and do not provide information on any other topics
        - Please strictly deny if the user asks to assist with any other topic regarding the document files and say that you can only assist regarding document files
        - Always ask the user in the beginning weather he wants to continue in English or Hindi. Strictly continue the further chat in the laguage selected by the user
        Thank you for your assistance and do remember to communicate with user strictly only regarding the files they have given as input and no other than that.
        """
        messages=[  
        {'role':'system', 'content': prompt},    
        {'role':'user', 'content':user_input}  ]
        response=get_completion_from_messages(messages, temperature=1)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    # else:
    #     user_input=update.message.text

    #     prompt="""You are a friendly Document Assistant.
    #     Your job is to help users find specific information from their documents or summarize them. 
    #     You will take the document file as an input and output the required content in a structured format. 
    #     You can communicate with the user, but strictly only regarding the files they have given as input and no other than that. 
    #     Please follow these instructions to fulfill your role:
    #     - Ask the user to upload their document and specify the type of document it is (PDF, Excel, Word, PPT, etc.)
    #     - Ask the user what information they are looking for or if they would like you to summarize the document for them
    #     - Provide the user with the information they need in a structured format such as JSON or HTML
    #     - Check whether all conditions are satisfied before providing the output
    #     - Only assist the user with the documents that they have provided as input and do not provide information on any other topics
    #     - Please strictly deny if the user asks to assist with any other topic regarding the document files and say that you can only assist regarding document files
    #     - Always ask the user in the beginning weather he wants to continue in English or Hindi. Strictly continue the further chat in the laguage selected by the user
    #     Thank you for your assistance and do remember to communicate with user strictly only regarding the files they have given as input and no other than that.
    #     """
    #     messages=[  
    #     {'role':'system', 'content': prompt},    
    #     {'role':'user', 'content':update.message.text}  ]
    #     response=get_completion_from_messages(messages, temperature=1)
    #     await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

# Handle Document input

async def doc_Inputs(update: Update, context: CallbackContext):
    #loading the document
    user_input = update.message
    if user_input.document:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Processing the document...")
        loader = TelegramChatFileLoader(user_input)
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

        user_input=update.message
        if user_input.text: 
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

            question=update.message.text 
            result=qa_chain({"query": question})
            await context.bot.send_message(chat_id=update.effective_chat.id, text=result)
    elif user_input.text:

        user_input=update.message.text

        prompt="""You are a friendly Document Assistant.
        Your job is to help users find specific information from their documents or summarize them. 
        You will take the document file as an input and output the required content in a structured format. 
        You can communicate with the user, but strictly only regarding the files they have given as input and no other than that. 
        Please follow these instructions to fulfill your role:
        - Ask the user to upload their document and specify the type of document it is (PDF, Excel, Word, PPT, etc.)
        - Ask the user what information they are looking for or if they would like you to summarize the document for them
        - Provide the user with the information they need in a structured format such as JSON or HTML
        - Check whether all conditions are satisfied before providing the output
        - Only assist the user with the documents that they have provided as input and do not provide information on any other topics
        - Please strictly deny if the user asks to assist with any other topic regarding the document files and say that you can only assist regarding document files
        - Always ask the user in the beginning weather he wants to continue in English or Hindi. Strictly continue the further chat in the laguage selected by the user
        Thank you for your assistance and do remember to communicate with user strictly only regarding the files they have given as input and no other than that.
        """
        messages=[  
        {'role':'system', 'content': prompt},    
        {'role':'user', 'content':update.message.text}  ]
        response=get_completion_from_messages(messages, temperature=1)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)



# async def QandA(update: Update, context: CallbackContext):
#     # Build prompt
#     template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
#     {context}
#     Question: {question}
#     Helpful Answer:"""
#     QA_CHAIN_PROMPT=PromptTemplate.from_template(template)  

#     qa_chain=RetrievalQA.from_chain_type(
#     llm,
#     retriever=db.as_retriever(),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )

#     question=update.message.text 
#     result=qa_chain({"query": question})
#     await context.bot.send_message(chat_id=update.effective_chat.id, text=result)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("TOKEN", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # On different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))

    # On non-command i.e message - process the input based on its content type
    dp.add_handler(MessageHandler(filters.text | filters.document, doc_Inputs))


    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

    

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), start)
    application.add_handler(echo_handler)
    application.run_polling()
    



