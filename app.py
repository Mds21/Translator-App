import gradio as gr                   
from transformers import pipeline

translation_pipeline_german = pipeline('translation_en_to_de')

def hindi_translate(text_):
    # sentencepiece
    from transformers import MarianMTModel, MarianTokenizer

    model_name = "Helsinki-NLP/opus-mt-en-hi"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    english_text = text_

    inputs = tokenizer.encode(english_text, return_tensors="pt")

    translation = model.generate(inputs)

    hindi_translation = tokenizer.decode(translation[0], skip_special_tokens=True)
    return hindi_translation
    
def en_hi_translate(text):
    from googletrans import Translator
    translator = Translator()
    
    english_text = text
    
    translation = translator.translate(english_text, src='en', dest='hi')
    return translation.text
    

def translate_transformers(English,Language_To_Translate):
    if "German" in Language_To_Translate:
        results = translation_pipeline_german(English)
        return results[0]['translation_text']
    elif "Hindi" in  Language_To_Translate:
        results = en_hi_translate(English)
        return results
    

interface = gr.Interface(fn=translate_transformers, 
                         inputs=[gr.inputs.Textbox(lines=2, placeholder='Text to translate'),
                         gr.CheckboxGroup(["German", "Hindi"])],
                        outputs='text')

interface.launch()