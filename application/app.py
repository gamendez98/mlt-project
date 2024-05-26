import gradio as gr
from grammar_error_correction import GrammarErrorCorrector

from token_realignment import RobertaTokenWordAlignment
from token_realignment import BertTokenWordAlignment

def load_model(model_name):
    MODEL_PATH = f"../Project_Testing/{model_name}"
    if model_name.startswith("roberta"):
        TOKEN_ALIGNMENT = RobertaTokenWordAlignment()
    elif model_name.startswith("bert"):
        TOKEN_ALIGNMENT = BertTokenWordAlignment()
    return GrammarErrorCorrector(MODEL_PATH, TOKEN_ALIGNMENT)

# Initialize with a default model
gec = load_model("bert_ner_model")

def predict(text):
    return gec.correct_sentence(text)

def verify_text(input_text, model_name):
    global gec
    try:
        # Load the selected model
        gec = load_model(model_name)
        return gec.correct_sentence(input_text)
    except Exception as e:
        return f"Error al procesar el texto: {str(e)}"

def clear_text():
    return "", ""

custom_css = """
#header {
    text-align: left;
    margin-left: 10px;
}

#university-text {
    font-size: 20px;
    font-weight: bold;
}

#ml-text {
    font-size: 14px;
    color: #555;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div id='header'>
                    <div id='university-text'>Universidad de Los Andes</div>
                    <div id='ml-text'>Machine Learning Techniques. MISIS-4219</div>
                </div>
            """)
        with gr.Column():
            gr.Markdown("<h1 style='text-align: center;'>Spanish Grammar Verification Application</h1>")
        with gr.Column():
            pass
    with gr.Row():
        model_selector = gr.Dropdown(choices=["bert_ner_model", "roberta_ner_model"], value="bert_ner_model", label="Select Model")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=10, placeholder="Escribe o pega tu texto aquí", label="Introduzca texto")
        
        with gr.Column():
            result_output = gr.Textbox(lines=10, interactive=False, label="Resultado revisión")
    
    with gr.Row():
        check_btn = gr.Button("Verify")
        clean_btn = gr.Button("Clean")
    
    check_btn.click(fn=verify_text, inputs=[text_input, model_selector], outputs=result_output)
    clean_btn.click(fn=clear_text, inputs=None, outputs=[text_input, result_output])

demo.launch(server_name="0.0.0.0", server_port=7861)
# demo.launch(server_port=7681) # local check