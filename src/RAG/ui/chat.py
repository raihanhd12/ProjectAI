"""
Chat interface UI components for the RAG module.
"""
import time
import streamlit as st

from .. import db
from .. import models


def chat_component():
    """
    Component for chat interface.

    Returns:
        None
    """
    # Initialize RAG model if not already initialized
    if "rag_model" not in st.session_state:
        st.session_state.rag_model = models.RAGModel()

    # Initialize session state for query results
    if "query_results" not in st.session_state:
        st.session_state.query_results = None
    if "relevant_text" not in st.session_state:
        st.session_state.relevant_text = None
    if "relevant_text_ids" not in st.session_state:
        st.session_state.relevant_text_ids = None

    # Chat Section
    st.header("Chat with your Documents")
    st.divider()

    display_chat_history()
    handle_user_input()


def display_chat_history():
    """
    Display the chat history.

    Returns:
        None
    """
    # Get and display chat history
    chat_history = db.get_chat_history()

    if chat_history:
        for message in chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    # First show thinking process if available
                    if message.get("thinking_content"):
                        with st.expander("AI Thinking Process", expanded=True):
                            st.markdown(message["thinking_content"])

                    # Then display the main content
                    st.write(message["content"])

                    # Other expanders come after the content
                    if message.get("query_results"):
                        with st.expander("See retrieved documents"):
                            st.write(message["query_results"])

                    if message.get("relevant_text_ids"):
                        with st.expander("See most relevant document ids"):
                            st.write(message["relevant_text_ids"])

                    if message.get("relevant_text"):
                        with st.expander("See relevant text"):
                            st.write(message["relevant_text"])
    else:
        documents = db.get_documents()
        if documents:
            st.info("Ask a question about your documents using the input box below")
        else:
            st.warning(
                "Please upload and process documents in the Document Management section first")


def handle_user_input():
    """
    Handle user input and generate response.

    Returns:
        None
    """
    # Use st.chat_input for a cleaner chat interface
    prompt = st.chat_input("Ask a question about your documents...")

    # Process prompt when submitted
    documents = db.get_documents()
    if prompt and documents:
        # Save user message to DB
        db.save_message("user", prompt)

        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # Retrieve relevant documents
        retrieve_documents(prompt)

        # Generate assistant response
        generate_response(prompt)


def retrieve_documents(prompt):
    """
    Retrieve relevant documents for the given prompt.

    Args:
        prompt (str): User prompt

    Returns:
        bool: True if documents were retrieved successfully, False otherwise
    """
    # First, show a status while retrieving documents
    with st.status("Retrieving relevant documents...") as status:
        try:
            # Get results from vector DB
            results = st.session_state.rag_model.query_collection(prompt)
            st.session_state.query_results = results

            if results and len(results.get("documents", [[]])[0]) > 0:
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = st.session_state.rag_model.re_rank_documents(
                    prompt, context)

                # Save to session state
                st.session_state.relevant_text = relevant_text
                st.session_state.relevant_text_ids = relevant_text_ids

                status.update(
                    label="Documents retrieved successfully!", state="complete")
                return True
            else:
                st.error(
                    "No relevant documents found. Please try a different question or upload more documents.")
                status.update(
                    label="No relevant documents found", state="error")
                return False
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            status.update(label=f"Error: {str(e)}", state="error")
            return False


def generate_response(prompt):
    """
    Generate response from the LLM.

    Args:
        prompt (str): User prompt

    Returns:
        None
    """
    # Generate assistant response if we have relevant text
    if not st.session_state.relevant_text:
        return

    with st.chat_message("assistant"):
        # First, create and auto-expand the thinking process expander
        with st.expander("AI Thinking Process", expanded=True):
            thinking_placeholder = st.empty()

        # Next, create placeholder for the response below the thinking process
        response_placeholder = st.empty()

        # Variables to store response and thinking process
        full_response = ""
        thinking_content = ""
        in_thinking_section = False

        # Define thinking starter phrases
        thinking_starters = ["Okay", "Let me",
                             "I need to", "First", "Based on", "Looking at"]

        # Collect initial content to detect if it's a thinking process
        initial_content = ""
        initial_content_collected = False

        # Stream response from LLM
        for chunk in st.session_state.rag_model.call_llm(
            context=st.session_state.relevant_text,
            prompt=prompt
        ):
            # Handle explicit thinking tags
            if "<think>" in chunk:
                parts = chunk.split("<think>")
                if len(parts) > 0:
                    full_response += parts[0]
                if len(parts) > 1:
                    thinking_content += parts[1]
                in_thinking_section = True
                # We've found explicit tags, so stop checking for implicit thinking
                initial_content_collected = True

            elif "</think>" in chunk and in_thinking_section:
                parts = chunk.split("</think>")
                thinking_content += parts[0]
                if len(parts) > 1:
                    full_response += parts[1]
                in_thinking_section = False

            elif in_thinking_section:
                thinking_content += chunk

            else:
                # Check for implicit thinking starters if we haven't collected much content yet
                if not initial_content_collected and len(initial_content) < 100:
                    initial_content += chunk

                    # Check if we have enough content to determine if it's starting with thinking phrases
                    if len(initial_content) > 20 or "." in initial_content:
                        initial_content_collected = True

                        # Check if the initial content starts with any thinking starter phrases
                        if any(initial_content.strip().startswith(starter) for starter in thinking_starters):
                            # This looks like a thinking process without explicit tags
                            first_period = initial_content.find(".")
                            if first_period != -1 and first_period < 50:
                                # If there's a period early in the text, treat everything up to
                                # the first paragraph break as thinking
                                first_paragraph_break = initial_content.find(
                                    "\n\n")
                                if first_paragraph_break != -1:
                                    thinking_content = initial_content[:first_paragraph_break]
                                    full_response = initial_content[first_paragraph_break:]
                                else:
                                    # No paragraph break yet, put it all in thinking for now
                                    thinking_content = initial_content
                            else:
                                # No early period, just add it all to thinking for now
                                thinking_content = initial_content
                        else:
                            # Not a thinking starter, treat as normal response
                            full_response = initial_content
                else:
                    full_response += chunk

            # Update placeholders with latest content
            if in_thinking_section or not initial_content_collected:
                thinking_placeholder.markdown(thinking_content + "▌")
            else:
                thinking_placeholder.markdown(thinking_content)

            response_placeholder.markdown(
                full_response + ("" if in_thinking_section or not initial_content_collected else "▌"))

            # Add slight delay for typing effect
            time.sleep(0.01)

        # Final update to remove cursor
        thinking_placeholder.markdown(thinking_content)
        response_placeholder.markdown(full_response)

        # Debug expanders after the answer
        with st.expander("See retrieved documents"):
            st.write(st.session_state.query_results)

        with st.expander("See most relevant document ids"):
            st.write(st.session_state.relevant_text_ids)

        with st.expander("See relevant text"):
            st.write(st.session_state.relevant_text)

        # Save complete response with debug info to DB
        db.save_message(
            role="assistant",
            content=full_response,
            thinking_content=thinking_content,
            query_results=st.session_state.query_results,
            relevant_text_ids=st.session_state.relevant_text_ids,
            relevant_text=st.session_state.relevant_text
        )
