import gradio as gr
from grammar_error_correction import GrammarErrorCorrector

MODEL_PATH = "../Project_Testing/bert_ner_model"
gec = GrammarErrorCorrector(MODEL_PATH)

def predict(text):
    return gec.correct_sentence(text)

# Función para retornar el texto corregido haciendo uso del API
def verify_text(input_text):
    try:
        return gec.correct_sentence(input_text)
    except Exception as e:
        return f"Error al procesar el texto: {str(e)}"

# Definir una función para limpiar la entrada y la salida
def clear_text():
    return "", ""

# CSS personalizado para dar estilo al texto
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

# Crear la interfaz de Gradio
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
            gr.Markdown("<h1 style='text-align: center;'>Aplicativo verificación gramatical en español</h1>")
        with gr.Column():
            pass  # Esta columna vacía es para el espaciado en el HTML original

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=10, placeholder="Escribe o pega tu texto aquí", label="Introduzca texto")
        
        with gr.Column():
            result_output = gr.Textbox(lines=10, interactive=False, label="Resultado revisión")
    
    with gr.Row():
        check_btn = gr.Button("Verificar")
        clean_btn = gr.Button("Limpiar")
    
    check_btn.click(fn=verify_text, inputs=text_input, outputs=result_output)
    clean_btn.click(fn=clear_text, inputs=None, outputs=[text_input, result_output])

# Lanzar la aplicación Gradio
demo.launch(server_name="0.0.0.0", server_port=7861)


