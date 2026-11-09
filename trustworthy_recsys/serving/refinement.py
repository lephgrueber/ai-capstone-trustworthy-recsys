"""Typed, bounded natural-language preference interpretation."""

from __future__ import annotations

from typing import Protocol
import re

from pydantic import ValidationError

from trustworthy_recsys.serving.contracts import (RefinementRequest, RefinementResponse,
    UnavailableRefinement, AppliedRefinement, ClarificationRefinement, UnsupportedRefinement, PreferenceState)


class RefinementInterpreter(Protocol):
    name: str
    version: str

    def interpret(self, request: RefinementRequest) -> RefinementResponse: ...


class UnavailableInterpreter:
    name = "unavailable"
    version = "none"

    def interpret(self, request: RefinementRequest) -> RefinementResponse:
        return UnavailableRefinement(
            message="Natural-language refinement is not configured. Structured preference controls remain available."
        )


class CatalogInterpreter:
    """Bounded English parser: genre requests and exact catalog-title seeds.

    This never fabricates a title, embeds arbitrary text, or treats similarity
    scores as probabilities. Unrecognized clauses are returned for clarification.
    """
    name = 'catalog_rules'
    version = '1.0'

    def __init__(self, catalog):
        self.catalog = catalog
        self.titles = {}
        for movie in catalog.movies.values():
            title = re.sub(r'\s*\(\d{4}\)$', '', movie.title).strip()
            aliases = [movie.title, title]
            match = re.match(r'(.+), (The|A|An)$', title)
            if match:
                natural_title = f'{match[2]} {match[1]}'
                aliases.append(natural_title)
                aliases.append(natural_title + movie.title[len(title):])
            for alias in aliases:
                self.titles.setdefault(alias.casefold(), set()).add(movie.movie_id)

    def interpret(self, request):
        text = request.text.strip().casefold().replace('’', "'")
        if re.search(r'\bnot only\b', text):
            return UnsupportedRefinement(message='Please list the genres to include or avoid directly; no changes were applied.')
        if any(item not in self.catalog.supported_ids for item in
               request.preferences.history_items + request.preferences.excluded_movie_ids):
            return UnsupportedRefinement(message='Some saved movie choices are not in the active catalog. Reset those choices first.')
        if text in {'reset', 'reset preferences', 'clear preferences', 'start over'}:
            return AppliedRefinement(preference_revision=request.preference_revision+1,
                preferences=PreferenceState(), summary='Cleared movie preferences.')
        # A title clause consumes its suffix so words inside a title are not genre commands.
        title_match = re.search(r'\b(?:similar to|movies like|films like|like)\s+(.+)$', text)
        next_state = request.preferences.model_dump()
        descriptions = []
        resolved_movies = []
        if title_match:
            title = title_match.group(1).strip(' \".!?')
            found = self.titles.get(title, set())
            if not found:
                return ClarificationRefinement(question='I could not match that title. Use its exact catalog title, with the year if needed.')
            if len(found) != 1:
                return ClarificationRefinement(question='That title matches multiple movies. Include the release year.')
            movie_id = next(iter(found))
            if re.search(r"\b(?:not|don't|do not|without|avoid)\b", text[:title_match.start()]):
                return ClarificationRefinement(question='For a movie to avoid, use the hide control. “Movies like” adds a positive example.')
            if movie_id in next_state['excluded_movie_ids']:
                return ClarificationRefinement(question='That movie is currently hidden. Unhide it before using it as a positive example.')
            if movie_id not in next_state['history_items']:
                next_state['history_items'].append(movie_id)
            descriptions.append('Using '+self.catalog.movies[movie_id].title+' as a positive example')
            resolved_movies.append(self.catalog.movies[movie_id])
            text = text[:title_match.start()]
        aliases = {g.casefold(): g for g in self.catalog.genres}
        science_fiction = 'Sci-Fi' if 'Sci-Fi' in self.catalog.genres else 'Science Fiction'
        children = 'Children' if 'Children' in self.catalog.genres else 'Family'
        for alias, canonical in {'sci-fi': science_fiction, 'sci fi': science_fiction,
                                 'science fiction': science_fiction, 'comedies': 'Comedy',
                                 'romantic': 'Romance', 'documentaries': 'Documentary',
                                 'animated': 'Animation', 'kids': children}.items():
            if canonical in self.catalog.genres:
                aliases[alias] = canonical
        pattern = r'(?<!\w)('+'|'.join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))+r')(?!\w)'
        recognized = []
        for clause in re.split(r'[,;.!?]|\bbut\b', text):
            matches = list(re.finditer(pattern, clause))
            excluded = False
            end = 0
            for match in matches:
                between = clause[end:match.start()]
                if re.search(r"\b(?:no|not|without|avoid|exclude|don't|do not)\b", between):
                    excluded = True
                if re.search(r'\b(?:include|want|prefer)\b', between) and not re.search(r"\b(?:not|don't|do not)\b", between):
                    excluded = False
                genre = aliases[match.group()]
                recognized.append((genre, excluded))
                end = match.end()
            remainder = re.sub(pattern, ' ', clause)
            remainder = re.sub(r"\b(?:show|me|find|recommend|want|would|please|some|a|an|the|i|to|watch|movies|movie|films|film|and|or|with|without|no|not|avoid|exclude|include|prefer|don't|do|any|only)\b", ' ', remainder)
            if remainder.strip(' :-\"'):
                return UnsupportedRefinement(message='I can search by genre or “movies like [exact title]”. Plot, actor, year filters, and mood requests are not supported yet; no changes were applied.')
        for genre, excluded in recognized:
            if (genre, not excluded) in recognized:
                return ClarificationRefinement(question=f'Should I include or avoid {genre}? Your request says both.')
            add = 'exclude_genres' if excluded else 'include_genres'
            remove = 'include_genres' if excluded else 'exclude_genres'
            next_state[remove] = [g for g in next_state[remove] if g != genre]
            if genre not in next_state[add]:
                next_state[add].append(genre)
            descriptions.append(('Avoiding ' if excluded else 'Including ')+genre)
        if not descriptions:
            return UnsupportedRefinement(message='Try “show comedy without horror” or “movies like [exact title]”.')
        try:
            preferences = PreferenceState.model_validate(next_state)
        except ValidationError:
            return UnsupportedRefinement(message='This request exceeds the saved-preference limits. Remove some choices first.')
        return AppliedRefinement(preference_revision=request.preference_revision+1,
                                 preferences=preferences, summary='; '.join(descriptions)+'.',
                                 resolved_movies=resolved_movies)
