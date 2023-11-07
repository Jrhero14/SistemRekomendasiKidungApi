from django.contrib import admin
from django.urls import path
from ninja import NinjaAPI
from pydantic import BaseModel
from .utils import Rekomendasi

api = NinjaAPI()

class Body(BaseModel):
    query: str

@api.get("/")
def root(request):
    return {"message": "Rekomendasi Lagu Pujian Kristen Menggunakan Metode TS-SS"}


@api.post("/rekomendasi/")
def rekomendasi(request, requestBody: Body):
    Query = requestBody.query
    RekomendasiLagu = Rekomendasi()
    hasil = RekomendasiLagu.GetRecomendation(query=Query, valueSimilarity=True)
    response = {
        'rekomendasi':[]
    }
    for idx, i in hasil.iterrows():
        response['rekomendasi'].append(
            {
                'nomor': str(i['Nomor']),
                'judul': str(i['Judul']),
                'similarity': "{: 2f}".format(i['Similarity']),
            }
        )
    return response

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', api.urls),
]