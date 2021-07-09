import argparse
from typing import List

from flask import Flask, request, jsonify
import pickle, os
from genre.trie import Trie, MarisaTrie
from genre.fairseq_model import mGENRE
from rust_fst import Map


class GENREModelHandler(object):

    def __init__(self, path: str, load_mapping: bool, no_gpu: bool):
        self.no_gpu = no_gpu
        self.trie = self._load_pickled(path, "titles_lang_all105_marisa_trie_with_redirect.pkl")
        self.mapping: Map = Map(os.path.join(path, "lang_title2wikidataID.fst"))
        self.model = self._load_model(path)

    def _load_pickled(self, path: str, filename: str):
        with open(os.path.join(path, filename), "rb") as input_file:
            return pickle.load(input_file)

    def _load_model(self, path: str):
        model_path = os.path.join(path, "fairseq_multilingual_entity_disambiguation")
        model = mGENRE.from_pretrained(model_path)
        if not self.no_gpu: model.cuda()
        model.eval()
        return model

    def _allow_token(self, batch_id, sent):
        return [e for e in self.trie.get(sent.tolist()) if e < len(self.model.task.target_dictionary)]

    def _text_to_id(self, name: str):
        key = "+".join(reversed(name.split(" >> ")))
        val = self.mapping[key]
        return f"Q{val}"

    def predict(self, sentences: List[str]):
        results = self.model.sample(sentences,
            prefix_allowed_tokens_fn=self._allow_token,
            text_to_id=self._text_to_id,
            marginalize=True
        )
        for result in results:
            for entity in result:
                entity['scores'] = entity['scores'].tolist()
                entity['score'] = entity['score'].tolist()
        return results


def create_app():
    app = Flask(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--load-mapping", type=bool, default=False)
    parser.add_argument("--no-gpu", type=bool, default=False)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    handler = GENREModelHandler(args.data_dir, args.load_mapping, args.no_gpu)

    @app.route("/", methods=["POST"])
    def predict():
        data: any = request.json
        inputs = data.get("inputs")
        inputs = [inputs] if isinstance(inputs, str) else inputs
        output = handler.predict(inputs)
        return jsonify({"inputs": inputs, "outputs": output})

    return app, args


if __name__ == '__main__':
    app, args = create_app()
    app.run(host=args.host, port=args.port, threaded=False)