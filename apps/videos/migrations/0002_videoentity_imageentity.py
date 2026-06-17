"""Add account-scoped VideoEntity + ImageEntity (ported from ezvision)."""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("videos", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="VideoEntity",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("account_id", models.CharField(max_length=255)),
                ("video_url", models.CharField(blank=True, max_length=1024, null=True)),
                ("index_name", models.CharField(blank=True, max_length=255, null=True)),
                ("sas_url", models.URLField(max_length=500)),
                ("file_name", models.CharField(blank=True, max_length=255, null=True)),
                ("status", models.CharField(choices=[("Initialized", "Initialized"), ("Processing", "Processing"), ("Completed", "Completed"), ("Canceled", "Canceled"), ("Reserved", "Reserved")], default="Initialized", max_length=20)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("modified", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name="ImageEntity",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("account_id", models.CharField(max_length=255)),
                ("video_url", models.CharField(blank=True, max_length=1024, null=True)),
                ("index_name", models.CharField(max_length=255)),
                ("sas_url", models.CharField(max_length=1024)),
                ("description", models.TextField(blank=True, max_length=4096, null=True)),
                ("timestamp", models.TimeField(blank=True, null=True)),
                ("location", models.CharField(blank=True, max_length=255)),
                ("status", models.CharField(blank=True, max_length=255)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("modified", models.DateTimeField(auto_now=True)),
                ("video", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="images", to="videos.videoentity")),
            ],
        ),
    ]
