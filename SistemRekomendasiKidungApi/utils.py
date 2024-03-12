import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from nltk.tokenize import word_tokenize
import re


# TS-SS Similarity
class TS_SS_Similarity:
    # Cosine similarity
    def Cosine(self, document1_vector, document2_vector):
        dot_product = np.dot(document1_vector, document2_vector.T)
        denominator = (np.linalg.norm(document1_vector) * np.linalg.norm(document2_vector))
        return dot_product / denominator

    # Euclidean distance
    def Euclidean(self, document1_vector, document2_vector):
        vec1 = document1_vector.copy()
        vec2 = document2_vector.copy()
        if len(vec1) < len(vec2): vec1, vec2 = vec2, vec1
        vec2 = np.resize(vec2, (vec1.shape[0], vec1.shape[1]))
        return np.linalg.norm(vec1 - vec2)

    # angle between two vectors
    def Theta(self, document1_vector, document2_vector):
        return np.arccos(self.Cosine(document1_vector, document2_vector)) + np.radians(10)

    # triangle formed by two vectors and ED as third side
    def Triangle(self, document1_vector, document2_vector):
        theta = np.radians(self.Theta(document1_vector, document2_vector))
        return ((np.linalg.norm(document1_vector) * np.linalg.norm(document2_vector)) * np.sin(theta)) / 2

    # difference in magnitude of two vectors
    def Magnitude_Difference(self, vec1, vec2):
        return abs((np.linalg.norm(vec1) - np.linalg.norm(vec2)))

    # sector area similarity
    def Sector(self, document1_vector, document2_vector):
        ED = self.Euclidean(document1_vector, document2_vector)
        MD = self.Magnitude_Difference(document1_vector, document2_vector)
        theta = self.Theta(document1_vector, document2_vector)
        return math.pi * (ED + MD) ** 2 * theta / 360

    # function which is acivated on call
    def __call__(self, document1_vector, document2_vector):
        result = self.Triangle(document1_vector, document2_vector) * self.Sector(document1_vector, document2_vector)
        return result


class Rekomendasi:
    def __init__(self):
        self.vectorizer = self._loadTfIdf_vectorizer()
        self.tf_idf = self._loadTfIdf_result()
        self.tf_idf_query = None
        self.queryuser = ''
        self.SimilartyTS_SS = TS_SS_Similarity()

    def _loadTfIdf_vectorizer(self):
        with open('data/tfidf_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        return vectorizer

    def _loadTfIdf_result(self):
        with open('data/tfidf_result.pkl', 'rb') as file:
            tfidfResult = pickle.load(file)
        return tfidfResult

    def Preprocessing(self):
        soupQuerySong = pd.DataFrame(data={
            'soupQuery': [self.queryuser]
        })

        # Menghapus tanda baca pada soup
        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        # Menghapus angka pada soup
        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].apply(lambda x: re.sub(r'[1-9]', '', str(x)))
        # Menghapus spasi ganda menjadi spasi tunggal pada soup
        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].apply(lambda x: re.sub('\s+', ' ', x))
        # Menghapus spasi di awal dan akhir pada soup
        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].apply(lambda x: str(x).lstrip().strip())

        # Tokenisasi Menggunakan NLTK
        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].apply(lambda x: word_tokenize(x))

        # Stopwords bahasa indonesia
        indonesia_stopwords = stopwords.words('indonesian')

        # Fungsi stopword removal
        def stopwords_removal(words):
            return [word for word in words if word not in indonesia_stopwords]

        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].apply(stopwords_removal)

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # menerapkan stemmer ke dataframe
        def get_stemmed_term(document):
            return [stemmer.stem(text=word) for word in document]

        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].apply(get_stemmed_term)

        # mengubah list pada kolom soup menjadi string
        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].agg(lambda x: ' '.join(map(str, x)))

        self.tf_idf_query = self.vectorizer.transform(soupQuerySong['soupQuery'])

    def GetRecomendation(self, query: str, valueSimilarity: bool = False):
        self.queryuser = query
        self.Preprocessing()

        obj_similarityQuery = {}
        temp = []
        for idx1, embed1 in enumerate(self.tf_idf):
            temp.append(self.SimilartyTS_SS((embed1).toarray(), (self.tf_idf_query).toarray()).mean())
            obj_similarityQuery[f'Nilai'] = temp

        soupSongs = pd.read_excel('data/soup.xlsx')
        judulLagu = pd.read_excel('data/KoleksiLagu.xlsx')
        df_similarityTS_SS = pd.DataFrame(
            obj_similarityQuery,
            columns=['Nilai']
        )
        df_similarityTS_SS['Judul'] = judulLagu['Judul'].tolist()
        df_similarityTS_SS['Nomor'] = soupSongs['nomor'].tolist()

        if valueSimilarity:
            return df_similarityTS_SS.sort_values(['Nilai'])[:10]
        else:
            return df_similarityTS_SS['Nilai'].sort_values().index.tolist()[:10]