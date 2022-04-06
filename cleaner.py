from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import emoji
import string
import re 
import unicodedata

# Function to clean the data
def clean_data (tweet):
    """    
    Input is datatype 'str': tweet (noisy tweet)
    Output is datatype 'str': tweet (cleaned tweet)
    """
    #Convert each emoji to text
    tweet = emoji.demojize(tweet)
    #convert to lowercase
    tweet = tweet.lower()
    #remove punctuation
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)
    #remove hashtags
    tweet = re.sub(r'#\w*','', tweet)
    #remove mentions 
    tweet = re.sub('@[\w]*','',tweet)
    #remove urls
    tweet = re.split('https:\/\/.*', str(tweet))[0]
    #remove emojis 
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emojis
        u"\U0001F300-\U0001F5FF"  # symbols 
        u"\U0001F680-\U0001F6FF"  # maps
        u"\U0001F1E0-\U0001F1FF"  # flags
                           "]+", flags = re.UNICODE)
    tweet = regrex_pattern.sub(r'', tweet)    
    #remove numbers 
    tweet = re.sub(r'\d+','', tweet)    
    #remove acsii
    tweet = unicodedata.normalize('NFKD', tweet).encode('ascii', 'ignore').decode('utf-8')
    #remove extra whitespaces 
    tweet = re.sub(r'\s\s+', ' ', tweet)
    #remove space in front of tweet
    tweet = tweet.lstrip(' ')
    return tweet



# Function to replace weblinks with text web_url
def handle_weblinks(text):
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    match = re.findall(pattern_url, text)
    
    for sub in match:
        text = re.sub(pattern_url, 'web_url', text)
    return text.strip().lower()

# Define function to help tokenize the data
def tokenize(text):
    tokenizer = TreebankWordTokenizer()
    
    return tokenizer.tokenize(text)

# Function to help out with either of stemming or lemmatization
def transform(text_list, method='lemma'):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    words = []
    if method == 'lemma':
        for word, tag in pos_tag(text_list):
            word_tag = tag[0].lower()
            word_tag = word_tag if word_tag in ['a', 'r', 'n', 'v'] else None
            if not word_tag:
                lemma = word
            else:
                lemma = lemmatizer.lemmatize(word, word_tag)
            
            words.append(lemma)

    elif method == 'stem':
        for word in text_list:
            stem = stemmer.stem(word)
            words.append(stem)
            
    else:
        return(f"ERROR: '{method}' is an unknown transformation method use 'stem' or 'lemma'")

    return words