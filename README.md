### GENRE-docker

Docker container with simple REST API for [mGENRE](https://github.com/facebookresearch/GENRE) multilingal entity disambiguation model.

Run container on GPU (requires nvidia-docker2):
```shell
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -p 8080 -v $(pwd)/data:/root/data genre:latest
```

Run container without GPU:
```shell
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8080 -v $(pwd)/data:/root/data genre:latest python3 run_server.py --no-gpu True
```