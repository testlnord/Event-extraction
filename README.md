# Event-extraction
Project aim is to extract events from news articles. Now it's just bunch of heuristics
on top of [spacy](https://spacy.io/) grammar trees. We are working on it.

Bellow is short description of this project. Full project documentation absent, but we
 hope to make it someday.

### What is event

We assume that event is a triple of (Subject, Action, Object). For example

 ```todo add example```

# What can be done now

### News collection and extraction

We have some crawlers written. So you can run

```python3 spacy_event_extractor.py ```

to get about a couple of thousands articles from different websites. This script will
also extract events just after it and put everything to database.

### Running web-interface

We have a simple web-UI to view and manipulate extracted events. You can start it just
by running:

```python3 run_web_server.py start```

This command will start small server on port 9090.
It can be accessed only from localhost.


# Installation

We haven't made an install script yet, so it's all just a sequence of manual operations.

1. **Dependencies**

   You can install everything only by running

   ```pip install -r requirements.txt```

   We are trying to keep it up to date.

   *Note: we highly recommend to keep everything in separate virtualenv.*

2. **Database creation**

   We are using postgres as our DBMS and depend on it.
   So you should have installed version of postgres.
   I have 9.3.14 installed on my system and everything seems to work

   ```psql postgres -f db/create.sql```

    MacOS psql path:

    ```/Applications/Postgres.app/Contents/Versions/9.5/bin/psql postgres -f create.sql```

3. **Config creation**

    Just copy ```default_config.cfg``` to ```config.cfg``` and set actual values to
    all fields.

    *Note: DO NOT modify ```default_config.cfg``` as it can be changed in the future
    and you can lose your settings.*


