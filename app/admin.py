from django.contrib import admin
from .models import Project, UserProfile, ModelAccuracy  # ✅ Import Profile

# Register your models here.
admin.site.register(Project)
admin.site.register(UserProfile)  # ✅ Register Profile
admin.site.register(ModelAccuracy)


