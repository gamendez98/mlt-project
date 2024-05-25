import gradio as gr

from grammar_error_correction.grammar_error_correction import GrammarErrorCorrector
from grammar_error_correction.token_realignment import RobertaTokenWordAlignment

MODEL_PATH = 'model/roberta_ner_model'
gec = GrammarErrorCorrector(MODEL_PATH, RobertaTokenWordAlignment())

# MODEL_PATH = 'model/bert_ner_model'
# gec = GrammarErrorCorrector(MODEL_PATH, BertTokenWordAlignment())


def predict(text):
    return gec.correct_sentence(text)


demo = gr.Interface(
    fn=predict,
    inputs='text',
    outputs='text',
)

if __name__ == '__main__':
    demo.launch()
