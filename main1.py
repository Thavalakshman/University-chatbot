import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
import numpy as np

# Load data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

class ChatbotApp:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot Interface")

        self.chatbox = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=100, height=20)
        self.chatbox.pack(padx=100, pady=100)

        self.user_input = tk.Entry(master, width=150)
        self.user_input.pack(pady=50)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack()

        # Initialize chatbot components
        self.initialize_chatbot()

    def initialize_chatbot(self):
        # Include your chatbot initialization code here
        # For example, load the model and other necessary components
        pass

    def send_message(self):
        user_message = self.user_input.get()
        self.display_message(f"You: {user_message}")

        # Use your existing chatbot functions to get a response
        bot_response = self.get_bot_response(user_message)

        self.display_message(f"Chatbot: {bot_response}\n")
        self.user_input.delete(0, tk.END)

    def get_bot_response(self, user_message):
        # Use your existing predict_class and get_response functions here
        bow = self.bag_of_words(user_message)
        res = model.predict(np.array([bow]))[0]
        error_threshold = 0.24
        results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

        if return_list:
            tag = return_list[0]['intent']
            list_of_intents = intents['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    return result
        return "I'm sorry, I don't understand."




    def bag_of_words(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        bag = [0] * len(words)
        for i, word in enumerate(words):
            if word in sentence_words:
                bag[i] = 1
        return np.array(bag)

    def display_message(self, message):
        self.chatbox.insert(tk.END, message + "\n")
        self.chatbox.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
