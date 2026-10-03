import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { acquireToken, revokeToken, writeGuardSummary } from './publisher-auth.mjs'

const env = { ACTIONS_ID_TOKEN_REQUEST_URL: 'https://oidc.example/token?existing=1', ACTIONS_ID_TOKEN_REQUEST_TOKEN: 'request-fixture' }
const response = (body, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => body })

test('publisher requests a separate masked Claude token with read-only contents access', async () => {
  const calls = []
  const masks = []
  const request = async (url, init) => {
    calls.push({ url: String(url), ...init })
    return calls.length === 1 ? response({ value: 'oidc-fixture' }) : response({ token: 'claude-fixture' })
  }
  assert.equal(await acquireToken(env, request, (value) => masks.push(value)), 'claude-fixture')
  assert.deepEqual(masks, ['oidc-fixture', 'claude-fixture'])
  assert.equal(new URL(calls[0].url).searchParams.get('audience'), 'claude-code-github-action')
  assert.equal(new URL(calls[0].url).searchParams.get('existing'), '1')
  assert.deepEqual(JSON.parse(calls[1].body).permissions, { contents: 'read', pull_requests: 'write', issues: 'write' })
  assert.equal(calls[1].headers.Authorization, 'Bearer oidc-fixture')
})

test('upstream workflow-validation refusal cannot fall back to the workflow token', async () => {
  let calls = 0
  const directory = mkdtempSync(join(tmpdir(), 'claude-auth-summary-'))
  const summaryPath = join(directory, 'summary.md')
  try {
    await assert.rejects(
      acquireToken({ ...env, GH_TOKEN: 'fallback' }, async () =>
        ++calls === 1
          ? response({ value: 'oidc-fixture' })
          : response({ error: { details: { error_code: 'workflow_not_found_on_default_branch' } } }, 403),
      ),
      (error) => {
        assert.match(error.message, /blocked by upstream workflow validation/)
        writeGuardSummary(error, summaryPath)
        return true
      },
    )
    const summary = readFileSync(summaryPath, 'utf8')
    assert.match(summary, /No review was run/)
    assert.match(summary, /no alternative publishing identity/)
    writeGuardSummary(new Error('HTTP 500'), summaryPath)
    assert.equal(readFileSync(summaryPath, 'utf8'), summary, 'generic errors are not mislabeled as workflow guards')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
  assert.equal(calls, 2)
})

test('legacy app_token response is accepted and invalid/missing tokens are rejected', async () => {
  for (const body of [{}, { token: 'bad\nworkflow-command' }, { token: null }]) {
    let calls = 0
    await assert.rejects(
      acquireToken(env, async () => (++calls === 1 ? response({ value: 'oidc' }) : response(body))),
      /no valid app token/,
    )
  }
  let calls = 0
  assert.equal(
    await acquireToken(env, async () => (++calls === 1 ? response({ value: 'oidc' }) : response({ app_token: 'legacy' }))),
    'legacy',
  )
})

test('exchange errors expose status, never provider response details', async () => {
  let calls = 0
  await assert.rejects(
    acquireToken(env, async () => (++calls === 1 ? response({ value: 'oidc' }) : response({ message: 'sensitive-provider-detail' }, 500))),
    (error) => {
      assert.match(error.message, /HTTP 500/)
      assert.doesNotMatch(error.message, /sensitive/)
      return true
    },
  )
})

test('missing OIDC environment fails before any network call', async () => {
  await assert.rejects(
    acquireToken({}, () => assert.fail('network called')),
    /requires GitHub Actions OIDC/,
  )
})

test('malformed JSON cannot expose response fragments through parser errors', async () => {
  for (const failedCall of [1, 2]) {
    let calls = 0
    await assert.rejects(
      acquireToken(env, async () =>
        ++calls === failedCall
          ? {
              ok: true,
              json: async () => {
                throw new SyntaxError('sensitive-response-fragment')
              },
            }
          : response({ value: 'oidc' }),
      ),
      (error) => {
        assert.match(error.message, /invalid JSON response/)
        assert.doesNotMatch(error.message, /sensitive/)
        return true
      },
    )
  }
})

test('status-only token requests checks permission without comment permission', async () => {
  const calls = []
  await acquireToken({ ...env, CLAUDE_PUBLISH_STATUS_ONLY: 'true' }, async (url, init) => {
    calls.push(init)
    return calls.length === 1 ? response({ value: 'oidc' }) : response({ token: 'status' })
  })
  assert.deepEqual(JSON.parse(calls[1].body).permissions, { contents: 'read', checks: 'write' })
})

test('cleanup revokes only the publisher token and reports failures', async () => {
  let call
  await revokeToken({ CLAUDE_PUBLISH_TOKEN: 'publisher' }, async (url, init) => {
    call = { url, ...init }
    return response({}, 204)
  })
  assert.equal(call.url, 'https://api.github.com/installation/token')
  assert.equal(call.method, 'DELETE')
  assert.equal(call.headers.Authorization, 'Bearer publisher')
  await revokeToken({}, () => assert.fail('network called'))
  await revokeToken({ CLAUDE_PUBLISH_TOKEN: 'expired' }, async () => response({}, 401))
  await assert.rejects(
    revokeToken({ CLAUDE_PUBLISH_TOKEN: 'publisher' }, async () => response({}, 500)),
    /revocation failed/,
  )
})
