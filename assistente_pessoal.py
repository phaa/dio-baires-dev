from transformers import pipeline
import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import os

class AssistenteVozPreTreinado:
    def __init__(self):
        # Inicializa o sintetizador de voz
        self.engine = pyttsx3.init()
        
        # Carrega o modelo Vosk para reconhecimento de voz em português
        if not os.path.exists("model-pt"):
            exit(1)
        
        self.model = Model("model-pt")
        self.rec = KaldiRecognizer(self.model, 16000)
        
        # Configura o PyAudio
        self.p = pyaudio.PyAudio()
        
        # Carrega o modelo de linguagem BERTimbau
        print("Carregando modelo de linguagem...")
        self.nlp = pipeline(
            "text2text-generation",
            model="pierreguillou/gpt2-small-portuguese",
            tokenizer="pierreguillou/gpt2-small-portuguese"
        )
        
        print("Assistente pronto!")
    
    def ouvir(self):
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )
        
        print("Ouvindo...")
        
        while True:
            data = stream.read(4000)
            if len(data) == 0:
                break
                
            if self.rec.AcceptWaveform(data):
                resultado = json.loads(self.rec.Result())
                if resultado.get("text", ""):
                    stream.stop_stream()
                    stream.close()
                    return resultado["text"]
        
        stream.stop_stream()
        stream.close()
        return ""
    
    def processar_comando(self, texto):
        # Adiciona um contexto para melhorar as respostas
        contexto = f"""
        Você é um assistente virtual útil e amigável.
        Usuário: {texto}
        Assistente:"""
        
        # Gera resposta usando o modelo
        resposta = self.nlp(contexto, 
                          max_length=100,
                          num_return_sequences=1,
                          do_sample=True,
                          temperature=0.7)[0]['generated_text']
        
        # Extrai apenas a resposta do assistente
        resposta = resposta.split("Assistente:")[-1].strip()
        
        return resposta if resposta else "Desculpe, não entendi."
    
    def falar(self, texto):
        self.engine.say(texto)
        self.engine.runAndWait()
    
    def executar(self):
        print("Assistente iniciado! Diga algo...")
        
        try:
            while True:
                comando = self.ouvir()
                if comando:
                    print(f"Você disse: {comando}")
                    
                    if "sair" in comando.lower():
                        print("Encerrando assistente...")
                        break
                    
                    resposta = self.processar_comando(comando)
                    print(f"Assistente: {resposta}")
                    self.falar(resposta)
        
        finally:
            self.p.terminate()


if __name__ == "__main__":
    assistente = AssistenteVozPreTreinado()
    assistente.executar()