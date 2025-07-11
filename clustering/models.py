from django.db import models

# Create your models here.
# models.py
from django.db import models

class ClusterSnapshot(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    json_data = models.TextField()  # stores the clustered DataFrame in JSON

    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d')})"
