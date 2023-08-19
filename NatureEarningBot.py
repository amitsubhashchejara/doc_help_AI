# IMPORTING ALL THE NEEDED MODULES AND LIBRARIES
import asyncio
import telegram
from telegram import Update
import os
import openai
from telegram.ext import Updater, CommandHandler, CallbackContext, ApplicationBuilder, ContextTypes, MessageHandler, filters, ApplicationBuilder, CallbackQueryHandler
import tracemalloc
from telegram import CallbackQuery
tracemalloc.start()

#GETTING ALL THE API'S 

from dotenv import load_dotenv, find_dotenv
_=load_dotenv(find_dotenv())

openai.api_key=os.getenv('api_key_openai_1')


bot = telegram.Bot(os.getenv('T_bot_api_1'))
TOKEN=os.getenv('T_bot_api_1')
BOT_ID=5831895868
application = ApplicationBuilder().token(TOKEN).build()

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response=openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    # print(str(response.choices[0].message))
    return response.choices[0].message["content"]

#  To send the first message to the user

async def start(update: Update, context: CallbackContext):
    user = update.effective_user  # Get the user object
    user_id = user.id

    if 'greeted' not in context.user_data:
        # This user has not been greeted yet
        context.user_data['greeted'] = True
        
        sticker_id = "CAACAgIAAxkBAAEf2kJkPWxJeC0zSz1kOizkm-BcpKQO5gACAQEAAladvQoivp8OuMLmNC8E"
        await update.message.reply_sticker(sticker_id)
        
        await update.message.reply_text('HELLO !!!\nWelcome to NatureEarning')
        await update.message.reply_text('Start talking to me to get your queries resolved.')
    else:
        user_input = update.message.text
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
        Thank you for your assistance and do remember to communicate with user strictly only regarding the files they have given as input and no other than that.
        """
        messages=[  
        {'role':'system', 'content': prompt},    
        {'role':'user', 'content':update.message.text}  ]
        response=get_completion_from_messages(messages, temperature=1)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    
async def reply(update: Update, context: CallbackContext):
    user_input = update.message.text
    messages=[  
    {'role':'system', 'content':'You are friendly chatbot.'},    
    {'role':'user', 'content':update.message.text}  ]
    response=get_completion_from_messages(messages, temperature=1)
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), start)
    application.add_handler(echo_handler)
    application.run_polling()
    



