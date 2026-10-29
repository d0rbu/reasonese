import assert from 'node:assert/strict';
import test from 'node:test';
import { originalCohortUrl } from '../viewer/app.js';

test('original cohort link accepts only site-local archive directories', () => {
  assert.equal(originalCohortUrl('/kimi-original-cohort-20261029/'), '/kimi-original-cohort-20261029/');
  assert.equal(originalCohortUrl('/archives/original/'), '/archives/original/');
  for (const value of [undefined, null, '', 'https://example.com/', '//example.com/',
    'javascript:alert(1)', '/\\example.com/', '/../original/', '/%2e%2e/original/', '/original/?next=external']) {
    assert.equal(originalCohortUrl(value), null, String(value));
  }
});
