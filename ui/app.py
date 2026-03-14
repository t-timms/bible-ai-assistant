"""
Gradio Web UI for Bible AI Assistant: text and voice chat.
Expects RAG server at localhost:8081, Kokoro TTS at localhost:8880.
"""
import gradio as gr

# TODO: Implement per guide Section 18:
# - transcribe(audio_path) with Faster-Whisper (large-v3-turbo, cuda, float16)
# - synthesize(text) via POST to TTS_URL
# - chat(message, history) via POST to RAG_URL with messages
# - voice_chat(audio, history): transcribe -> chat -> synthesize -> return history + audio
# - gr.Blocks with Chatbot, Tab(Text) with Textbox.submit, Tab(Voice) with Audio stop_recording -> voice_chat
# - demo.launch(server_name='0.0.0.0', server_port=7860)

def placeholder_chat(message: str, history: list) -> str:
    return "Bible AI UI not yet wired. Start RAG server and implement app.py per guide Section 18."


with gr.Blocks(title="Bible AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Bible AI Assistant")
    gr.Markdown("Ask about Scripture via text or voice.")
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="Ask a question...")
    msg.submit(
        lambda m, h: (placeholder_chat(m, h), ""),
        [msg, chatbot],
        [chatbot, msg],
    )
    gr.Examples(
        ["What does John 3:16 mean?", "Who was the Apostle Paul?", "What does the Bible say about forgiveness?"],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
