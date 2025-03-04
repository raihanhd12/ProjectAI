import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class TextSummarizer:
    def __init__(self, model_name="google/pegasus-xsum", device=None, cache_dir=None):
        """
        Initialize the text summarizer with a Hugging Face model.

        Args:
            model_name: Name of the Hugging Face summarization model
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            cache_dir: Custom directory to store downloaded models
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer with custom cache directory
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=cache_dir)

        # Move model to appropriate device
        self.model.to(self.device)

        # Create summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def summarize(self, text, max_length=150, min_length=30):
        """
        Summarize the provided text using the Hugging Face model.

        Args:
            text: The text to summarize
            max_length: Maximum length of the summary in tokens
            min_length: Minimum length of the summary in tokens

        Returns:
            The summarized text
        """
        if not text:
            return "Please provide text to summarize."

        try:
            # Handle long text by chunking
            if len(text) > 1024:
                result = self._chunk_summarize(text, max_length, min_length)
            else:
                # Direct summarization for shorter text
                result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                result = result[0]['summary_text']

            return result

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _chunk_summarize(self, text, max_length=150, min_length=30):
        """
        Summarize long text by breaking it into chunks, summarizing each chunk,
        and then summarizing the combined results.

        Args:
            text: The long text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary

        Returns:
            The summarized text
        """
        # Split text into sentences
        sentences = text.split('. ')

        # Create chunks of sentences
        max_chunk_size = 1000  # characters
        chunks = []
        current_chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            sentence = sentence.strip() + '. '
            sentence_size = len(sentence)

            if current_chunk_size + sentence_size > max_chunk_size:
                # Add current chunk to chunks list
                chunks.append(''.join(current_chunk))
                # Start a new chunk
                current_chunk = [sentence]
                current_chunk_size = sentence_size
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_chunk_size += sentence_size

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(''.join(current_chunk))

        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarizer(
                chunk,
                max_length=max(50, max_length // 2),
                min_length=min(20, min_length // 2),
                do_sample=False
            )
            chunk_summaries.append(summary[0]['summary_text'])

        # Combine chunk summaries
        combined_summary = ' '.join(chunk_summaries)

        # If combined summary is still long, summarize it again
        if len(combined_summary) > 1000:
            final_summary = self.summarizer(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return final_summary[0]['summary_text']

        return combined_summary


class HuggingFaceModels:
    """Helper class to provide information about available summarization models."""

    @staticmethod
    def get_recommended_models():
        """Get a list of recommended summarization models."""
        return [
            {
                "name": "google/pegasus-xsum",
                "description": "State-of-the-art model for extreme summarization (very concise)"
            },
            {
                "name": "facebook/bart-large-cnn",
                "description": "Good for news article summarization (more detailed)"
            },
            {
                "name": "sshleifer/distilbart-cnn-12-6",
                "description": "Faster, distilled version of BART (good balance)"
            },
            # {
            #     "name": "philschmid/distilbart-cnn-12-6-samsum",
            #     "description": "Optimized for dialogue and conversation summarization"
            # },
            # {
            #     "name": "t5-small",
            #     "description": "Lightweight T5 model, good for limited resources"
            # }
        ]
