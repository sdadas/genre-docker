FROM nvcr.io/nvidia/pytorch:21.06-py3
COPY requirements.txt requirements.txt
COPY examples /root/examples
COPY run_server.py /root/run_server.py
RUN apt update && apt install unzip && pip install -r requirements.txt
WORKDIR /root/
CMD ["python", "run_server.py"]