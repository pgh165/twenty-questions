FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 레이어 캐시 최적화: requirements.txt만 먼저 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (개발 환경에서는 compose의 bind mount가 덮어씀)
COPY . .

# SQLite DB 볼륨 마운트 지점
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
