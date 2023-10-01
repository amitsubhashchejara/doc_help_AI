# IMPORTING ALL THE NEEDED MODULES AND LIBRARIES

# all the telegram bot modules
import pathlib
import tempfile
from telegram import Update
import os
from telegram.ext import (
    filters,
    CommandHandler,
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
    ApplicationBuilder,
)


# all the openai modules
import openai

# all the langchain modules
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TelegramChatFileLoader, TelegramChatApiLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA


# GETTING ALL THE ESSENTIAL API'S

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("api_key_openai_1")
TOKEN = os.getenv("T_bot_api_2")


llm_name="gpt-3.5-turbo"
llm=ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=os.getenv('api_key_openai_1'))

BOT_ID = os.getenv("BOTID2")

# DEFINING DIFFERENT FUNCTIONS FOR EVERY LIBRARY


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    # print(str(response.choices[0].message))
    return response.choices[0].message["content"]


# TO SEND THE FIRST MESSAGE TO THE USER


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "greeted" not in context.user_data:
        # This user has not been greeted yet
        context.user_data["greeted"] = True
        sticker_id = (
            "CAACAgIAAxkBAAEf2kJkPWxJeC0zSz1kOizkm-BcpKQO5gACAQEAAladvQoivp8OuMLmNC8E"
        )
        await update.message.reply_sticker(sticker_id)
        await update.message.reply_text("HELLO !!!\nWelcome to DocHelpAI")
        await update.message.reply_text(
            "Please click on menu button and select /doc_help to\nget help related to document or just chat with me\nin general here."
        )
    else:
        chat_query = update.message.text
        prompt="""
                You are a friendly Document Assistant. Your job is to help users find specific information from their documents or summarize them. You will take the document file as an input and output the required content in a structured format. You can communicate with the user, but strictly only regarding the files they have given as input and no other than that. Please follow these instructions to fulfill your role:
                - Ask the user to click on /doc_help command from the menu button and to upload their document and specify the type of document it is (PDF, Excel, Word, PPT, etc.)
                - Ask the user what information they are looking for or if they would like you to summarize the document for them
                - Provide the user with the information they need in a structured format such as JSON or HTML
                - Check whether all conditions are satisfied before providing the output and make sure that you assist with doucment related task only, say no to questions related to any other topic except document related questions.
                - Strictly only assist the user with the documents related task, the documents that they have provided as input and do not provide information on any other topics
                Thank you for your assistance!"""
        messages=[  
            {'role':'system', 'content':prompt},    
            {'role':'user', 'content':chat_query}  ]

        chat_result=get_completion_from_messages(messages, temperature=0.5)
        await update.message.reply_text(chat_result)


async def doc_help(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a PDF file to get started. I will process it and you "
        "can ask me questions related to it. Other document formats will be supported in the future."
    )


async def chat_with_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("document_uploaded", False):
        chat_query = update.message.text
        prompt="""
                You are a friendly Document Assistant. Your job is to help users find specific information from their documents or summarize them. You will take the document file as an input and output the required content in a structured format. You can communicate with the user, but strictly only regarding the files they have given as input and no other than that. Please follow these instructions to fulfill your role:
                - Ask the user to click on /doc_help command from the menu button and to upload their document and specify the type of document it is (PDF, Excel, Word, PPT, etc.)
                - Ask the user what information they are looking for or if they would like you to summarize the document for them
                - Provide the user with the information they need in a structured format such as JSON or HTML
                - Check whether all conditions are satisfied before providing the output and make sure that you assist with doucment related task only, say no to questions related to any other topic except document related questions.
                - Strictly only assist the user with the documents related task, the documents that they have provided as input and do not provide information on any other topics
                Thank you for your assistance!"""
        messages=[  
            {'role':'system', 'content':prompt},    
            {'role':'user', 'content':chat_query}  ]

        chat_result=get_completion_from_messages(messages, temperature=0.5)
        await update.message.reply_text(chat_result)
        return

    text = update.effective_message.text
    # TODO: Process the text here:
    # Build prompt
    
    template = """Use the following pieces of context to answer the question at the end. Make sure that you 
    only answer the questions which are related to the document provided to you.If the question is out of the 
    context or out of the topic like other than the document related task, just say that please ask questions related to this doucment. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    question = text
    result = qa_chain({"query": question})
    await context.bot.send_message(chat_id=update.effective_chat.id, text=result['result'])


async def download_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc_file = await update.effective_message.effective_attachment.get_file()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        path = pathlib.Path(tmp_dir_name) / str(update.effective_user.id)
        file = await doc_file.download_to_drive(path)

        # ------------------
        # TODO: Process the document here:
        await process_and_load_pdf(update, context, file)

        context.user_data["document_uploaded"] = True


async def process_and_load_pdf(
    update: Update, context: ContextTypes.DEFAULT_TYPE, pdf_file_path: pathlib.Path
):
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Processing the document...",
    )
    loader=PyPDFLoader(str(pdf_file_path))
    pages=loader.load()

    # splitting the document
    c_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=300, separator=" "
    )
    splitted_pages = c_splitter.split_documents(pages)

    # embedding the document
    embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
    global db
    db = DocArrayInMemorySearch.from_documents(
        splitted_pages,
        embedding=embedding,
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Document processed successfully!!!\nYou can ask me questions related to this document now.",
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Ask me questions related to this document",
    )


def main():
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("doc_help", doc_help))
    application.add_handler(MessageHandler(filters.Document.PDF, download_pdf))
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), chat_with_bot)
    )
    application.run_polling()


if __name__ == "__main__":
    # echo_handler_1 = MessageHandler(filters.TEXT & (~filters.COMMAND), start)
    # application.add_handler(echo_handler_1)
    main()
