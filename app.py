"""
Gradio Frontend for BNM Policy RAG Agent.
Designed for Hugging Face Spaces deployment.
"""

import os
import gradio as gr
from src.graph import create_graph

# Initialize the agent graph
agent = create_graph()

def query_agent(question: str, history: list) -> str:
    """
    Process a question through the RAG agent and return the response.
    """
    if not question.strip():
        return "Please enter a question about BNM policies."
    
    try:
        # Initialize state
        initial_state = {
            "original_query": question,
            "current_query": question,
            "retry_count": 0,
            "revision_count": 0,
            "sub_questions": [],
            "documents": [],
            "generation": "",
            "filters": {},
            "grade_status": "",
            "hallucination_status": "",
            "critique": ""
        }
        
        # Run the agent
        final_state = None
        for state in agent.stream(initial_state):
            final_state = state
        
        # Extract the answer
        if final_state:
            response = list(final_state.values())[0].get("generation", "")
            if response:
                return response
        
        return "I couldn't find relevant information in the BNM policy documents."
        
    except Exception as e:
        return f"Error processing your question: {str(e)}"


# Example questions for users
EXAMPLE_QUESTIONS = [
    "What are the e-KYC requirements under BNM regulations?",
    "What is the maximum late payment charge for credit cards?",
    "What are the Simplified CDD requirements?",
    "What is the minimum age requirement for a debit card?",
    "What are the technology risk management requirements under RMiT?",
]

# Build the Gradio interface
with gr.Blocks(title="BNM Policy RAG Agent") as demo:
    
    gr.Markdown(
        """
        # 🏦 BNM Policy RAG Agent
        
        Ask questions about **Bank Negara Malaysia (BNM)** regulatory policy documents.
        This agent uses advanced retrieval and verification to provide accurate, cited answers.
        
        ---
        """,
        elem_classes=["header-text"]
    )
    
    # Chatbot component - will auto-detect format
    chatbot = gr.Chatbot(
        label="Conversation",
        height=450
    )
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What are the e-KYC requirements under BNM regulations?",
            lines=2,
            scale=4,
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)
    
    with gr.Accordion("📝 Example Questions", open=False):
        gr.Markdown("Click on any example to use it:")
        for example in EXAMPLE_QUESTIONS:
            gr.Button(example, size="sm").click(
                fn=lambda x=example: x,
                outputs=question_input
            )
    
    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
    
    gr.Markdown(
        """
        ---
        
        ⚠️ **Disclaimer**: This tool is for informational purposes only and does not constitute 
        official legal or financial advice. Always verify with official documents on 
        [bnm.gov.my](https://www.bnm.gov.my/policy-documents).
        """
    )
    
    # Event handlers - using message dict format for Gradio 6.0
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        
        # Ensure chat_history is a list
        if chat_history is None:
            chat_history = []
        
        bot_response = query_agent(message, chat_history)
        
        # Append in message format
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        
        return "", chat_history
    
    submit_btn.click(
        fn=respond,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )
    
    question_input.submit(
        fn=respond,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )
    
    clear_btn.click(fn=lambda: [], outputs=chatbot)


if __name__ == "__main__":
    # Move CSS to launch() method for Gradio 6.0
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css="""
        .gradio-container {
            max-width: 900px !important;
            margin: auto;
        }
        .header-text {
            text-align: center;
            margin-bottom: 1rem;
        }
        """
    )