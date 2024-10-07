FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 5000

# 運行應用
CMD ["python", "main.py"]