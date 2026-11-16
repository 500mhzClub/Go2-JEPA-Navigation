"""Correct B004 output routing while retaining its original scientific admission."""
import importlib.util
import json
from pathlib import Path

from go2nav.research import digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/U013')
ORIGINAL = Path('outputs/gated/B004')
HELPERS = ('integration.py', 'run_one.py', 'check.py')


def need(value, message):
    if not value:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def inherited():
    record, _ = require_record('U013')
    verify_files(Path.cwd(), record['protocol']['source_sha256'])
    need(require_review(ORIGINAL/'software/review.json')['status'] == 'passed',
         'Original B004 software is not reviewed')
    software = json.loads((ORIGINAL/'software/result.json').read_text())
    need(software['research_sha256'] == require_record('B004')[1], 'B004 protocol changed')
    verify_files(ORIGINAL, software['source_sha256'])
    for field in ('repository_source_sha256', 'record_sha256', 'inherited_source_sha256'):
        verify_files(Path.cwd(), software[field])
    return module('u013_original_parents', ORIGINAL/'parents.py')


def qualified():
    inherited()
    _, rh = require_record('U013')
    review = require_review(BASE/'software/review.json')
    software = json.loads((BASE/'software/result.json').read_text())
    need(review['status'] == 'passed' and review['research_id'] == 'U013'
         and software['research_id'] == 'U013' and software['research_sha256'] == rh,
         'U013 routing software is not qualified')
    verify_files(BASE, software['source_sha256'])
    verify_files(Path.cwd(), software['record_sha256'])
    need(digest(BASE/'controlled_changes.json') == software['controlled_changes_sha256'],
         'Controlled routing changes differ')
    return dict(research_id='U013', research_sha256=rh,
                software_review_sha256=digest(BASE/'software/review.json'),
                source_sha256=software['source_sha256'])


def consultation(pre, expected):
    need(pre.get('u013_record_sha256') == expected['research_sha256']
         and pre.get('u013_software_review_sha256') == expected['software_review_sha256'],
         'Missing or stale U013 consultation binding')


def context(root, name):
    integration = qualified()
    ctx = inherited().context(root, name)
    consultation(ctx['consult'], integration)
    ctx['integration'] = integration
    return ctx
