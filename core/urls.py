from django.contrib import admin
from app import views
from django.urls import path, include
from app.views import (
    predict,
    index1,
    index,
    search,
    ticker,
    LoginUser,
    HomePage,
    news,
    LogoutUser,
    clicklogin,
    RegisterUser,
    ClickRegister,
    admin_dashboard
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app.urls')),  # Default home handled by views.home
    path('index1/', index1),
    path('index/', index),
    path('search/', search),
    path('predict/<str:ticker_value>/<str:number_of_days>/', predict),
    path('ticker/', ticker),
    path('loginuser/', LoginUser),
    path('homepage/', HomePage),
    path('news/', news),
    path('logout/', LogoutUser, name='logout'),
    path('clicklogin/', clicklogin),
    path('register_user/', RegisterUser),
    path('click_user/', ClickRegister),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('delete_user/<int:user_id>/', views.delete_user, name='delete_user'),
    path('active/', views.active_crypto, name='active_crypto'),


]
