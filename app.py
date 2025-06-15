import gradio as gr
from summarizer import summarize_text_input, summarize_pdf_file, summarize_url

def summarize(choice, text, pdf, url):
    if choice == "Text":
        return summarize_text_input(text)
    elif choice == "PDF":
        return summarize_pdf_file(pdf)
    elif choice == "URL":
        return summarize_url(url)
    else:
        return "Invalid input type"

iface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Radio(["Text", "PDF", "URL"], label="Input Type"),
        gr.Textbox(label="Enter Text", placeholder="Paste your text here"),
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Enter URL", placeholder="https://example.com")
    ],
    outputs="text",
    title="Text Summarizer",
    description="Summarizes Text, PDFs, and URLs using BART model"
)

if __name__ == "__main__":
    iface.launch()