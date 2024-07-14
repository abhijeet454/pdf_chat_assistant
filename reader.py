import pyttsx3
import time

def read_file(filename):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speech rate
    while True:
        with open(filename, 'r') as file:
            content = file.read()
            if content:
                engine.say(content)
                engine.runAndWait()
                # Clear the file content
                open(filename, 'w').close()
        time.sleep(1)

if __name__ == "__main__":
    read_file("answer.txt")
