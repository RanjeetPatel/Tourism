
FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD [
    "streamlit",
    "run",
    "tourism_project/deployment/app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0",
    "--server.enableXsrfProtection=false"
]
