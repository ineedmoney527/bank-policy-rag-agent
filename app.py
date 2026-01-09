"""
Gradio Frontend for Compliance-Aware Banking Agent.

A simple web UI for querying Malaysian banking regulations (BNM).
"""

import os
import logging
import gradio as gr

from src.graph import run_agent
from src.retriever import get_retriever
from src.ingest import run_ingestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_system():
    """Initialize the retriever on startup."""
    logger.info("Initializing Compliance-Aware Banking Agent...")
    retriever = get_retriever()
    try:
        retriever.initialize()
        logger.info("Retriever initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"Retriever not initialized (run ingestion first): {e}")
        return False


def query_agent(question: str, history: list) -> str:
    """
    Process a user query through the RAG agent.
    
    Args:
        question: User's question about banking regulations
        history: Chat history (not used currently)
    
    Returns:
        Agent's response with compliance information
    """
    if not question.strip():
        return "Please enter a question about Malaysian banking regulations."
    
    try:
        logger.info(f"Processing query: {question[:50]}...")
        result = run_agent(question)
        
        # Format the response
        response = result["final_response"]
        status = result.get("compliance_status", "unknown")
        iterations = result.get("iteration_count", 1)
        
        # Add metadata footer
        footer = f"\n\n---\n*Status: {status} | Iterations: {iterations}*"
        
        # Add sources if available
        if result.get("retrieved_docs"):
            sources = set()
            for doc in result["retrieved_docs"]:
                meta = doc.get("metadata", {})
                filename = meta.get("filename", "Unknown")
                sources.add(filename)
            
            if sources:
                footer += f"\n*Sources: {', '.join(sources)}*"
        
        return response + footer
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return f"❌ Error processing query: {str(e)}"


def ingest_documents() -> str:
    """Trigger document ingestion."""
    try:
        logger.info("Starting document ingestion...")
        run_ingestion()
        
        # Reinitialize retriever
        retriever = get_retriever()
        retriever._initialized = False
        retriever.initialize()
        
        return "✅ Documents ingested successfully! You can now ask questions."
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return f"❌ Ingestion failed: {str(e)}"


# Initialize on startup
is_ready = initialize_system()

# Create Gradio Interface
with gr.Blocks(
    title="BNM Compliance Agent",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        footer { display: none !important; }
    """
) as demo:
    
    gr.Markdown(
        """
        # 🏦 Malaysian Banking Compliance Agent
        
        Ask questions about **Bank Negara Malaysia (BNM)** regulations including:
        - AML/CFT policies
        - E-Money guidelines  
        - Claims settlement practices
        - Customer due diligence requirements
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
                show_copy_button=True,
                avatar_images=(None, "🤖"),
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the simplified CDD requirements for low-risk customers?",
                    scale=4,
                    show_label=False,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)
            
            clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ Clear Chat")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Controls")
            
            status_box = gr.Textbox(
                label="System Status",
                value="✅ Ready" if is_ready else "⚠️ Run ingestion first",
                interactive=False,
            )
            
            ingest_btn = gr.Button("📥 Ingest Documents", variant="secondary")
            ingest_output = gr.Textbox(
                label="Ingestion Status",
                interactive=False,
                visible=True,
            )
            
            gr.Markdown(
                """
                ---
                ### 📚 Sample Questions
                
                - What is the timeframe for BER claims?
                - Explain e-KYC requirements
                - What are STR reporting obligations?
                - Describe simplified CDD measures
                """
            )
    
    # Event handlers
    def respond(message, chat_history):
        bot_response = query_agent(message, chat_history)
        chat_history.append((message, bot_response))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    ingest_btn.click(ingest_documents, outputs=ingest_output)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
