from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200", api_key="UEttVF9aWUJ1Q0hTNl9FdzhWX3M6eDFKZUlLSEJDel94aWl3aHlENUVNdw==")

docs = [
    {"id": "1", "title": "Introduction to FastAPI", "content": "FastAPI is a modern web framework for Python."},
    {"id": "2", "title": "Elasticsearch Basics", "content": "Elasticsearch is a distributed search engine."},
    {"id": "3", "title": "Python Tips", "content": "Use list comprehensions for cleaner code."},
]

actions = [
    {
        "_index": "your_index",
        "_id": doc["id"],
        "_source": doc
    }
    for doc in docs
]

helpers.bulk(es, actions)
