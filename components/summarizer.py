import streamlit as st
from utils.chat_history import ChatHistory
from models.summarizer import HuggingFaceModels


def text_summarizer_app():
    """Text Summarizer application interface using Hugging Face models."""
    st.title("Text Summarizer")
    
    # Model selection
    recommended_models = HuggingFaceModels.get_recommended_models()
    model_options = [model["name"] for model in recommended_models]
    model_descriptions = {model["name"]: model["description"] for model in recommended_models}
    
    # Create columns for model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_model = st.selectbox(
            "Select model:",
            model_options,
            index=0,
            key="summarizer_model"
        )
    
    with col2:
        st.info(model_descriptions[selected_model])
    
    # Check if model needs to be loaded or changed
    if "summarizer_model_name" not in st.session_state or st.session_state.summarizer_model_name != selected_model:
        with st.spinner(f"Loading {selected_model}..."):
            # Import here to avoid loading transformers when not using this component
            from models.summarizer import TextSummarizer
            st.session_state.summarizer = TextSummarizer(
                model_name=selected_model,
                cache_dir="./models/summarization_cache" 
            )
            st.session_state.summarizer_model_name = selected_model
    
    # Text input section
    text_input = st.text_area("Enter or paste text to summarize:", height=300)
    
    # Parameters
    col1, col2 = st.columns([1,1])
    
    with col1:
        min_length = st.number_input(
            "Min length (tokens):", 
            min_value=10, 
            max_value=200, 
            value=30
        )
    
    with col2:
        max_length = st.number_input(
            "Max length (tokens):", 
            min_value=50, 
            max_value=500, 
            value=150
        )
    
    summarize_button = st.button("Summarize Text", use_container_width=True)
    
    if summarize_button and text_input:
        with st.spinner("Generating summary..."):
            summary = st.session_state.summarizer.summarize(
                text_input, 
                max_length=max_length,
                min_length=min_length
            )
        
        st.subheader("Summary:")
        st.write(summary)
        
        # Show token counts
        original_tokens = len(text_input.split())
        summary_tokens = len(summary.split())
        reduction = 100 - (summary_tokens / max(original_tokens, 1) * 100)
        
        st.caption(f"Original: ~{original_tokens} tokens | Summary: ~{summary_tokens} tokens | Reduction: {reduction:.1f}%")
        
        # Add to chat history
        ChatHistory.add_message(
            st.session_state.current_conversation_id,
            "user",
            f"[Summarize text with {selected_model}] Length: {len(text_input)} characters"
        )
        
        ChatHistory.add_message(
            st.session_state.current_conversation_id,
            "assistant",
            summary,
            {"model": selected_model, "reduction_percentage": f"{reduction:.1f}%"}
        )