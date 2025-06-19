from django.urls import path
from . import views
from .views import snapshot_list, snapshot_detail, export_snapshot_csv

urlpatterns = [
    path('', views.about, name='about'),
    path('upload/', views.upload_csv, name='upload'),
    path('results/', views.results, name='results'),
    path('clusters/<int:cluster_id>/', views.cluster_detail, name='cluster_detail'),
    path('about/', views.about, name='about'),
    path('plot/', views.plot, name='plot'),
    path('clean-data/', views.clean_data, name='clean_data'),
    path('upload-cleaned/', views.upload_cleaned, name='upload_cleaned'),
    path('cleaned-results/', views.cleaned_results, name='cleaned_results'),
    path('export-clustered/', views.export_clustered_csv, name='export_clustered'),
    path('export-cleaned/', views.export_cleaned_csv, name='export_cleaned'),
    path('plot/', views.plot, name='plot'),
    path('internal-cluster/', views.internal_cluster, name='internal_cluster'),
    path('demographics/', views.demographics_report, name='demographics_report'),
    path('form-predict/', views.form_predict, name='form_predict'),
    path('snapshots/', snapshot_list, name='snapshot_list'),
    path('snapshots/<int:snapshot_id>/', snapshot_detail, name='snapshot_detail'),
    path('snapshots/delete/<int:pk>/', views.delete_snapshot, name='delete_snapshot'),
    path('snapshots/export/<int:snapshot_id>/', export_snapshot_csv, name='export_snapshot_csv'),



]
