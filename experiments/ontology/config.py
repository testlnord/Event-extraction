
config = {
  # "global": {
  # }

  "base_dir": "/home/user/projects/Event-extraction",

  "models": {
    "dir": "experiments/models",

    "nlp": {
      # "name": "en_core_web_sm",
      "name": "en_core_web_md",
      "path": None,
      # "path": "models.v5.4.i5.epoch2",
    },

    # // "dbpedianet": "dbpedianet_model_noner.dr.noaug.v5.1.c3.all.inv_full_epoch04",
  },

  "ontology": {
    "endpoint": "http://localhost:8890/sparql-auth",
    "endpoint_user": "dba",
    "endpoint_passwd": "admin"
  },

  "data": {
    "articles_dir": "/home/user/datasets/dbpedia/articles3",
    "dir": "/home/user/datasets/dbpedia"

  }
}
