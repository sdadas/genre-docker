### GENRE-docker

Docker container with simple REST API for [mGENRE](https://github.com/facebookresearch/GENRE) multilingal entity disambiguation model.

Before running the container, create `data` directory with the following files:
- [Pre-trained mGENRE model](https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz)
- [Prefix trie with Wikipedia titles](http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl)
- [Titles to Wikidata IDs mapping](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EackTgD12FNPg7--nKOOPoIBDbY2T1GHe7e5cEwn9Xx_oA?e=W8RWP9)

Run container on GPU (requires nvidia-docker2):
```shell
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -p 8080 -v $(pwd)/data:/root/data genre:latest
```

Run container without GPU:
```shell
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8080 -v $(pwd)/data:/root/data genre:latest python3 run_server.py --no-gpu True
```
