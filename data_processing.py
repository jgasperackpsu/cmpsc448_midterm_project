import os
import re
import random
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize

# Download embedding data
nltk.download('punkt')