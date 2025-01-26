import nltk

# Download the necessary NLTK resources
nltk.download('punkt')      # For tokenization
nltk.download('stopwords')  # For stop words
nltk.download('wordnet')    # For lemmatization
nltk.download('punkt_tab')
import re
import html
import string
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords





# Load stop words
stop_words = stopwords.words('english')

def text_preprocess(text):
    # Remove special characters
    re1 = re.compile(r'  +')
    text = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    text = re1.sub(' ', html.unescape(text))

    # Remove non-ASCII characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Convert to lowercase
    text = text.lower()

    # Replace numbers
    text = re.sub(r'\d+', '', text)

    # Remove leading and trailing whitespaces
    text = text.strip()

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Lemmatize verbs
    words = ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

    return words

# Example usage
sample_text = "This is an example text with numbers 123 and special characters! #@&"
cleaned_text = text_preprocess(sample_text)
print(cleaned_text)