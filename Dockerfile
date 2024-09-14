FROM python:3.8.2-slim-buster

WORKDIR /workspace

COPY train.py /workspace/

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

CMD ["python3", "train.py"]