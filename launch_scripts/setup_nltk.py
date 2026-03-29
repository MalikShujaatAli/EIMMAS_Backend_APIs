import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Downloading NLTK punkt data...")
    nltk.download('punkt', quiet=False)
    # Also download punkt_tab which is sometimes required by newer NLTK versions
    nltk.download('punkt_tab', quiet=False)
    logger.info("Successfully downloaded NLTK punkt data.")
