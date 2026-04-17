from django.urls import path

from . import views

app_name = "game"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/start/", views.start_game, name="start_game"),
    path("api/answer/", views.answer, name="answer"),
]
