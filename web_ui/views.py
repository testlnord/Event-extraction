from flask import session, redirect, url_for, render_template, request, jsonify
from web_ui import app
from flask.ext.wtf import Form
from wtforms import IntegerField, DateField, SubmitField, TextAreaField
from db import DatabaseHandler
from event import Event
import datetime

DEFAULT_ARTICLES_COUNT = 10
db_handler = DatabaseHandler()


class EventsForm(Form):
    selected_event_id = -1
    publish_date = TextAreaField()
    entity1 = TextAreaField()
    action = TextAreaField()
    entity2 = TextAreaField()
    date = TextAreaField()
    sentence = TextAreaField()


class FetchArticleForm(Form):
    fetch_articles = SubmitField('Fetch new articles')


@app.route('/')
def redirect_to_events():
    return redirect(url_for('events'))


@app.route('/_load_events', methods=['POST'])
def load_events():
    session["start_index"] = session["current_index"]
    session["current_index"] += DEFAULT_ARTICLES_COUNT
    events = db_handler.get_events_starting_from(session["current_index"], datetime.datetime.now())
    return jsonify(result=[(db_handler.get_event_publish_date(e.id), db_handler.get_event_source(e.id).url, e.json()) for e in
                           events[session["start_index"]: session["current_index"]]])


@app.route('/_delete_event', methods=['POST'])
def delete_event_by_id():
    id = request.form.get('id', 0, type=int)
    db_handler.del_event_by_id(id)
    return jsonify(result=None)


@app.route('/_get_event', methods=['POST'])
def get_event_by_id():
    id = request.form.get('id', 0, type=int)
    event = db_handler.get_event_by_id(id)
    return jsonify(result=(db_handler.get_event_publish_date(event.id), db_handler.get_event_source(id).url, event.json()))


@app.route('/_join_events', methods=['POST'])
def join_events():
    ids = request.form.getlist('ids[]')
    join_entities1 = request.form.get('joinEntities1', 0, type=bool)
    join_actions = request.form.get('joinActions', 0, type=bool)
    join_entities2 = request.form.get('joinEntities2', 0, type=bool)

    db_handler.join_events(ids)

    if join_entities1:
        db_handler.join_entities_by_events(ids, "1")

    if join_actions:
        db_handler.join_actions_by_events(ids)

    if join_entities2:
        db_handler.join_entities_by_events(ids, "2")

    return jsonify(result=None)


def check_phrase(phrase, sentence):
    for word in phrase.split():
        if not word in sentence:
            return False
    return True


@app.route('/_modify_event', methods=['POST'])
def modify_event_by_id():
    event_id = request.form.get('id', 0, type=int)
    entity1 = request.form.get('entity1', 0, type=str)
    action = request.form.get('action', 0, type=str)
    entity2 = request.form.get('entity2', 0, type=str)

    entity1 = ' '.join(entity1.split())
    action = ' '.join(action.split())
    entity2 = ' '.join(entity2.split())

    #sentence = request.args.get('sentence', 0, type=str)
    sentence = db_handler.get_event_by_id(event_id).sentence

    if not check_phrase(entity1, sentence):
        return jsonify(result=None, error="Incorrect entity1!")
    if not check_phrase(action, sentence):
        return jsonify(result=None, error="Incorrect action!")
    if not check_phrase(entity2, sentence):
        return jsonify(result=None, error="Incorrect entity2!")

    db_handler.change_event(event_id, Event(entity1, entity2, action, sentence, None))

    event = db_handler.get_event_by_id(event_id)
    return jsonify(result=(db_handler.get_event_publish_date(event.id), event.json()), error=None)


@app.route('/events', methods = ['GET', 'POST'])
def events():
    form = EventsForm()
    session["start_index"] = session["current_index"] = 0
    return render_template("events.html", form = form)


@app.route('/sources', methods = ['GET', 'POST'])
def articles():
    print(request.method)
    articles = db_handler.get_sites()
    articles_forms = [FetchArticleForm(prefix=article[0]) for article in articles]
    for form, article in zip(articles_forms, articles):
        pass  # Todo: actually fetch articles from given source

    return render_template("sources.html", articles=zip(articles, articles_forms))


@app.route('/statistics', methods=['GET', 'POST'])
def statistics():
    return render_template("statistics.html", form=Form())

