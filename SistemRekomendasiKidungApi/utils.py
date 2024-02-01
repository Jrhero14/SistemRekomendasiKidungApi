import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

from nltk.tokenize import word_tokenize
import re


# TS-SS Similarity
class TS_SS:
    # Cosine similarity
    def Cosine(self, question_vector, sentence_vector):
        dot_product = np.dot(question_vector, sentence_vector.T)
        denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
        return dot_product / denominator

    # Euclidean distance
    def Euclidean(self, question_vector, sentence_vector):
        vec1 = question_vector.copy()
        vec2 = sentence_vector.copy()
        if len(vec1) < len(vec2): vec1, vec2 = vec2, vec1
        vec2 = np.resize(vec2, (vec1.shape[0], vec1.shape[1]))
        return np.linalg.norm(vec1 - vec2)

    # angle between two vectors
    def Theta(self, question_vector, sentence_vector):
        return np.arccos(self.Cosine(question_vector, sentence_vector)) + np.radians(10)

    # triangle formed by two vectors and ED as third side
    def Triangle(self, question_vector, sentence_vector):
        theta = np.radians(self.Theta(question_vector, sentence_vector))
        return ((np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector)) * np.sin(theta)) / 2

    # difference in magnitude of two vectors
    def Magnitude_Difference(self, vec1, vec2):
        return abs((np.linalg.norm(vec1) - np.linalg.norm(vec2)))

    # sector area similarity
    def Sector(self, question_vector, sentence_vector):
        ED = self.Euclidean(question_vector, sentence_vector)
        MD = self.Magnitude_Difference(question_vector, sentence_vector)
        theta = self.Theta(question_vector, sentence_vector)
        return math.pi * (ED + MD) ** 2 * theta / 360

    # function which is acivated on call
    def __call__(self, document1_vector, document2_vector):
        result = self.Triangle(document1_vector, document2_vector) * self.Sector(document1_vector, document2_vector)
        return result


class Rekomendasi:
    def __init__(self):
        self.soupSongs = pd.read_excel('data/soup.xlsx')
        self.vectorizer = TfidfVectorizer()
        self.tf_idf = self.vectorizer.fit_transform(self.soupSongs['soup'])
        self.tf_idf_query = None
        self.queryuser = ''
        self.SimilartyTS_SS = TS_SS()

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

        soupQuerySong['soupQuery'] = soupQuerySong['soupQuery'].swifter.apply(get_stemmed_term)

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