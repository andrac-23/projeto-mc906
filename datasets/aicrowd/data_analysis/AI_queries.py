import os
import google.generativeai as genai
from googletrans import Translator, constants
import time

GOOGLE_API_KEY = SUA_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

translator = Translator()

# open classes.txt
classes = []
curr_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(curr_dir, "training", "yoloV5_annotations", "_classes.txt"), "r") as f:
    for line in f:
        curr_idx = line.split(": ")[0]
        curr_class = line.split(": ")[1].strip()
        while True:
            try:
                make_it_readable = "The following word is written in coding format: {}. Make it readable in English (with spaces and other grammatical constructions that make sense), and only answer the new readable version."
                readable_class = model.generate_content(make_it_readable.format(curr_class))
                readable_class = readable_class.text
                break
            except Exception as e:
                print("Erro ao fazer a pergunta. Tentando novamente em 60s...")
                print("Erro: {}".format(e))
                print("Preenchendo os dados até agora...")
                with open(os.path.join(curr_dir, "training", "yoloV5_annotations", "_classes_traduzidas.txt"), "w") as f:
                    for c in classes:
                        f.write(c + "\n")
                for i in range(60):
                    print("\r{}s left".format(60 - i), end=" ")
                    time.sleep(1)
        # print("Class: {} -> {}".format(curr_class, readable_class))
        translated_class = translator.translate(readable_class, src="en", dest="pt")
        translated_class = translated_class.text
        # print("Class: {} -> {}".format(readable_class, translated_class))
        classes.append("{}: {} ({})".format(curr_idx, translated_class, curr_class)) # class_id: translated_class (original_class)
        print("\r{}/322 done ({:.2f}%)".format(curr_idx, ((int(curr_idx) + 1) / 322) * 100), end=" ")

print("\nTodos os dados preenchidos! Salvando...")
with open(os.path.join(curr_dir, "training", "yoloV5_annotations", "_classes_traduzidas.txt"), "w") as f:
    for c in classes:
        f.write(c + "\n")
