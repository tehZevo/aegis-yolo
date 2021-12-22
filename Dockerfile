FROM python:3.8

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

#download
COPY download_models.py .
RUN python download_models.py

COPY . .

EXPOSE 80

CMD [ "python", "-u", "main.py" ]
