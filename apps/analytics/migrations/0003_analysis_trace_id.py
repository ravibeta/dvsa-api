"""Add Analysis.trace_id for commentary correlation (observability)."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analytics", "0002_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysis",
            name="trace_id",
            field=models.CharField(blank=True, db_index=True, default="", max_length=32),
        ),
    ]
