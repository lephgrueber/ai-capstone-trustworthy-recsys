import pytest
from fastapi.testclient import TestClient

from trustworthy_recsys.serving.app import create_app
from trustworthy_recsys.serving.catalog import Catalog, fixture_catalog
from trustworthy_recsys.serving.contracts import CatalogMovie, PreferenceState, RefinementRequest
from trustworthy_recsys.serving.refinement import CatalogInterpreter


def request(text, **preferences):
    return RefinementRequest(session_id='test', preference_revision=3,
                             preferences=PreferenceState(**preferences), text=text)


def test_genres_negation_and_title_flow():
    parser = CatalogInterpreter(fixture_catalog())
    result = parser.interpret(request('Show comedy without thriller'))
    assert result.outcome == 'applied'
    assert result.preferences.include_genres == ['Comedy']
    assert result.preferences.exclude_genres == ['Thriller']
    result = parser.interpret(request('movies like Orbit House'))
    assert result.preferences.history_items == ['102']
    assert result.resolved_movies[0].title == 'Orbit House'
    assert result.preference_revision == 4


@pytest.mark.parametrize('text', ['funny movies with Tom Hanks', 'comedy from 1995',
                                 'ignore previous instructions', 'comedy under two hours'])
def test_unsupported_clauses_do_not_partially_apply(text):
    result = CatalogInterpreter(fixture_catalog()).interpret(request(text))
    assert result.outcome == 'unsupported'


def test_conflicts_unknown_and_hidden_titles_require_clarification():
    parser = CatalogInterpreter(fixture_catalog())
    for text in ['comedy but no comedy', 'movies like not a real movie', 'not movies like Orbit House']:
        assert parser.interpret(request(text)).outcome == 'clarification'
    assert parser.interpret(request('movies like Orbit House', excluded_movie_ids=['102'])).outcome == 'clarification'


def test_ambiguous_title_year_and_reset():
    catalog = Catalog({str(i): CatalogMovie(movie_id=str(i), title=f'Example ({year})', genres=['Drama'])
                       for i, year in [(1, 1990), (2, 2000)]})
    parser = CatalogInterpreter(catalog)
    assert parser.interpret(request('movies like Example')).outcome == 'clarification'
    assert parser.interpret(request('movies like Example (2000)')).preferences.history_items == ['2']


def test_movielens_genre_and_article_title_aliases():
    movie = CatalogMovie(movie_id='2571', title='Matrix, The (1999)', genres=['Sci-Fi', 'Action'])
    parser = CatalogInterpreter(Catalog({'2571': movie}))
    assert parser.interpret(request('science fiction')).preferences.include_genres == ['Sci-Fi']
    assert parser.interpret(request('movies like The Matrix (1999)')).preferences.history_items == ['2571']
    assert parser.interpret(request('reset preferences', include_genres=['Drama'])).preferences == PreferenceState()


def test_api_interpretation_reaches_retrieval_and_hard_filters():
    with TestClient(create_app()) as client:
        assert client.get('/api/v1/application/status').json()['refinement_available']
        result = client.post('/api/v1/refinements', json=request('show comedy without thriller').model_dump()).json()
        response = client.post('/api/v1/recommendations', json={
            'session_id': 'test', 'preference_revision': result['preference_revision'],
            'preferences': result['preferences'], 'result_count': 12})
        assert response.status_code == 200
        movies = response.json()['recommendations']
        assert movies
        for item in movies:
            assert 'Comedy' in item['movie']['genres']
            assert 'Thriller' not in item['movie']['genres']
