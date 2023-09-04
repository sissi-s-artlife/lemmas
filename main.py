import nltk
import ssl
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Suppress NLTK download messages
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize the WordNet lemmatizer and load the list of stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Define a list of common English determiners
determiners = ["the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"]

# Define a function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if POS tag not found

# Sample text
text = input("Insert the text you want to lemmatize and clean the determiners off: ")

# Tokenize the text into words
words = nltk.word_tokenize(text)

# Remove stop words and determiners
filtered_words = [word for word in words if word.lower() not in stop_words and word.lower() not in determiners]

# Assign POS tags to the filtered words
pos_tags = nltk.pos_tag(filtered_words)

# Lemmatize each word with the appropriate POS tag
lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

# Join the lemmatized words back into a sentence
lemmatized_text = ' '.join(lemmatized_words)

# Print the lemmatized text
print("Original text:", text)
print("Lemmatized text (after removing stop words and determiners):", lemmatized_text)
