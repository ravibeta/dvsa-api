# DVSA API - Drone Video Sensing Analytics API

A professional-grade Django REST API for drone video sensing and analytics, built with modern best practices.

## 🎯 Project Overview

DVSA (Drone Video Sensing Analytics) is a comprehensive API platform for processing, analyzing, and managing drone video data with AI/ML capabilities including computer vision and geospatial analysis.

## 🏗️ Architecture

- **Framework**: Django 5.2.4 with Django REST Framework 3.16.0
- **Database**: PostgreSQL (configurable for SQLite in development)
- **Authentication**: Token-based with JWT support
- **API Standards**: RESTful with proper versioning
- **Async Support**: Celery for background tasks
- **Monitoring**: Comprehensive logging and error tracking

## 📋 Prerequisites

- Python 3.10+
- PostgreSQL 12+
- Redis (for Celery)
- Docker & Docker Compose (optional)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/ravibeta/dvsa-api.git
cd dvsa-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate