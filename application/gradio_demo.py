import gradio_demo as gr

from grammar_error_correction.grammar_error_correction import GrammarErrorCorrector

MODEL_PATH = 'model/bert_ner_model'
gec = GrammarErrorCorrector(MODEL_PATH)


def predict(text):
    return gec.correct_sentence(text)


demo = gr.Interface(
    fn=predict,
    inputs='text',
    outputs='text',
)

demo.launch()
