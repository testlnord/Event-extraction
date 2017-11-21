import os.path


config = {
    "base_dir": "/home/user/projects/Event-extraction",

    "models": {
        "dir": "/home/user/projects/Event-extraction/experiments/models",

        "nlp": {
            "dir": "/home/user/projects/Event-extraction/experiments/models/nlp",
            # "name": "en_core_web_sm",
            "name": "en_core_web_md",
            "path": None,
        },

        "dbpedianet": {

        }
    },

    "ontology": {
        "endpoint": "http://localhost:8890/sparql-auth",
        "endpoint_user": "dba",
        "endpoint_passwd": "admin"
    },

    "data": {
        "dir": "/home/user/datasets/dbpedia",
        "articles_dir": "/home/user/datasets/dbpedia/articles3",
        "rc_dir": "/home/user/datasets/dbpedia/rc",
        "ner_dir": "/home/user/datasets/dbpedia/ner"
    }
}


def load_nlp(model_name=None):
    import spacy

    nlp_conf = config['models']['nlp']
    if model_name:
        path = os.path.join(nlp_conf['dir'], model_name)
    else:
        path = nlp_conf['path']

    if path:
        return spacy.load('en', path=path)
    return spacy.load(nlp_conf['name'])
