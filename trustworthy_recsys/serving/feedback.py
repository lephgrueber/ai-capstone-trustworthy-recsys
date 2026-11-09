"""Durable, single-slot randomized assignments and capability-bound feedback."""

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contracts import FeedbackAssignment, RecommendationResponse

WINDOW = 1800
LATE_ALLOWANCE = 86400
POLICY = 'featured_epsilon_v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def probabilities(size, epsilon):
    if size < 1 or not 0 <= epsilon <= 1:
        raise ValueError('Invalid candidate count or exploration rate')
    return [1-epsilon+epsilon/size] + [epsilon/size]*(size-1)


class FeedbackError(ValueError):
    pass


class FeedbackStore:
    def __init__(self, path, mode='randomized', epsilon=.1, synthetic=False, provenance=None, clock=time.time):
        self.path, self.mode, self.epsilon = Path(path), mode, epsilon
        if mode not in {'observe', 'randomized'} or not 0 < epsilon <= 1:
            raise ValueError('Invalid feedback policy configuration')
        self.synthetic, self.provenance, self.clock = synthetic, provenance or {}, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (
                  id TEXT PRIMARY KEY, session TEXT NOT NULL, assignment_key TEXT NOT NULL,
                  request_hash TEXT NOT NULL, assigned REAL NOT NULL, payload TEXT NOT NULL,
                  UNIQUE(session, assignment_key));
                CREATE TABLE IF NOT EXISTS events (
                  id TEXT PRIMARY KEY, decision_id TEXT NOT NULL REFERENCES decisions(id),
                  received REAL NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_decision ON events(decision_id);
            ''')
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('secret', ?)", (secrets.token_hex(32),))
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('duplicate_events', '0')")
            self.secret = db.execute("SELECT value FROM metadata WHERE key='secret'").fetchone()[0].encode()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def signature(self, value):
        return hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    def response(self, payload):
        response = RecommendationResponse.model_validate(payload['response'])
        response.feedback = FeedbackAssignment(
            decision_id=payload['decision_id'], token=self.signature('decision:'+payload['decision_id']),
            assigned_at=datetime.fromtimestamp(payload['assigned'], timezone.utc).isoformat(),
            movie_id=payload['action'], propensity=payload['propensity'], policy=payload['policy'],
            eligible_count=len(payload['candidates']), synthetic=payload['synthetic'],
            age_ms=max(0, int((self.clock()-payload['assigned'])*1000)))
        return response

    def assign(self, request, response):
        # The input response contains the entire verified ranked pool. Freeze the
        # ordinary ranks 2..N as the grid, then randomize ONLY the separate hero.
        if not response.recommendations:
            return response
        session = self.signature('session:'+request.session_id)
        key = request.assignment_key or str(uuid.uuid4())
        request_hash = digest({'request': request.model_dump(exclude={'session_id', 'assignment_key'}),
                               'provenance': self.provenance, 'mode': self.mode, 'epsilon': self.epsilon,
                               'backend': response.backend_mode, 'synthetic': self.synthetic})
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT request_hash,payload FROM decisions WHERE session=? AND assignment_key=?', (session, key)).fetchone()
            if old:
                if old[0] != request_hash:
                    raise FeedbackError('Assignment key was reused with different context or policy')
                return self.response(json.loads(old[1]))
            ranked = response.recommendations
            grid = ranked[1:request.result_count]
            pool = [ranked[0], *ranked[request.result_count:]]
            epsilon = self.epsilon if self.mode == 'randomized' else 0.
            probs = probabilities(len(pool), epsilon)
            chosen = secrets.SystemRandom().choices(range(len(pool)), weights=probs, k=1)[0]
            response.recommendations = [pool[chosen], *grid]
            for rank, item in enumerate(response.recommendations, 1):
                item.rank = rank
            candidates = [{'movie_id': item.movie.movie_id, 'retrieval_score': item.retrieval_score,
                           'ranking_score': item.ranking_score} for item in pool]
            decision_id, assigned = str(uuid.uuid4()), self.clock()
            payload = {'schema_version': 1, 'decision_id': decision_id, 'assigned': assigned,
                'session': session, 'context': request.model_dump(exclude={'session_id', 'assignment_key'}),
                'backend': response.backend_mode, 'synthetic': self.synthetic or response.backend_mode == 'fixture',
                'provenance': self.provenance, 'policy': POLICY if epsilon else 'deterministic_observation_v1',
                'epsilon': epsilon, 'candidate_hash': digest(candidates), 'candidates': candidates,
                'slot': 'featured',
                'fixed_grid': [item.movie.movie_id for item in grid], 'action': pool[chosen].movie.movie_id,
                'action_index': chosen, 'propensity': probs[chosen], 'probabilities': probs,
                'ope_eligible': bool(epsilon and len(pool) > 1),
                'response': response.model_dump(mode='json', exclude={'feedback'})}
            db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?)',
                       (decision_id, session, key, request_hash, assigned, canonical(payload)))
        # No response leaves the service until the transaction is durable.
        return self.response(payload)

    def event(self, event):
        if not hmac.compare_digest(event.token, self.signature('decision:'+event.decision_id)):
            raise FeedbackError('Invalid decision token')
        body = event.model_dump(exclude={'token'})
        encoded, now = canonical(body), self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            decision = db.execute('SELECT payload FROM decisions WHERE id=?', (event.decision_id,)).fetchone()
            if not decision:
                raise FeedbackError('Unknown decision')
            payload = json.loads(decision[0])
            if event.movie_id != payload['action']:
                raise FeedbackError('Event movie is not the assigned featured movie')
            old = db.execute('SELECT payload FROM events WHERE id=?', (event.event_id,)).fetchone()
            if old:
                if old[0] != encoded:
                    raise FeedbackError('Event ID reused with different content')
                db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='duplicate_events'")
                return True
            if now > payload['assigned'] + WINDOW + LATE_ALLOWANCE:
                raise FeedbackError('Event arrived after the late-arrival allowance')
            if payload['assigned'] + event.elapsed_ms/1000 > now+5:
                raise FeedbackError('Event time is in the future')
            if event.kind == 'window_end' and event.elapsed_ms < WINDOW*1000:
                raise FeedbackError('Outcome window has not ended')
            count = db.execute('SELECT COUNT(*) FROM events WHERE decision_id=?', (event.decision_id,)).fetchone()[0]
            if count >= 512:
                raise FeedbackError('Decision event limit reached')
            db.execute('INSERT INTO events VALUES (?,?,?,?)', (event.event_id, event.decision_id, now, encoded))
        return False
