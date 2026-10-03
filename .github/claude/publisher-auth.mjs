// Dedicated publisher token using the pinned Claude action's OIDC protocol:
// https://github.com/anthropics/claude-code-action/blob/fa2b2666b747000bf42767d1f332065b375e3c8f/src/github/token.ts
// This is an upstream implementation dependency; verify it on action upgrades.
// The action revokes its own token before returning. This separate token is a
// masked step output, scoped to trusted writes, and revoked by always() cleanup.
import { appendFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const exchangeUrl = 'https://api.anthropic.com/api/github/github-app-token-exchange'
const timeout = () => AbortSignal.timeout(30000)
const validToken = (value) => typeof value === 'string' && /^[A-Za-z0-9_.-]+$/.test(value)
const readJson = async (response) => {
  try {
    const body = await response.json()
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error()
    return body
  } catch {
    // JSON parser errors can include unmasked fragments of the response body.
    throw new Error('Publisher authentication received an invalid JSON response.')
  }
}

export async function acquireToken(env, request = fetch, mask = () => {}) {
  if (!env.ACTIONS_ID_TOKEN_REQUEST_URL || !env.ACTIONS_ID_TOKEN_REQUEST_TOKEN) {
    throw new Error('Publisher authentication requires GitHub Actions OIDC (id-token: write).')
  }
  const url = new URL(env.ACTIONS_ID_TOKEN_REQUEST_URL)
  if (url.protocol !== 'https:') throw new Error('OIDC requires HTTPS.')
  url.searchParams.set('audience', 'claude-code-github-action')
  const oidcResponse = await request(url, {
    headers: { Authorization: `Bearer ${env.ACTIONS_ID_TOKEN_REQUEST_TOKEN}` },
    signal: timeout(),
  })
  if (!oidcResponse.ok) throw new Error(`OIDC request failed (HTTP ${oidcResponse.status}).`)
  const { value: oidc } = await readJson(oidcResponse)
  if (!validToken(oidc)) throw new Error('OIDC response did not contain a valid token.')
  mask(oidc)
  const permissions =
    env.CLAUDE_PUBLISH_STATUS_ONLY === 'true'
      ? { contents: 'read', checks: 'write' }
      : { contents: 'read', pull_requests: 'write', issues: 'write' }
  const response = await request(exchangeUrl, {
    method: 'POST',
    headers: { Authorization: `Bearer ${oidc}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ permissions }),
    signal: timeout(),
  })
  if (!response.ok) {
    const body = await readJson(response).catch(() => ({}))
    const guard =
      body.error?.details?.error_code === 'workflow_not_found_on_default_branch' ||
      (response.status === 401 &&
        [body.message, body.error?.message].some((message) => typeof message === 'string' && /workflow validation failed/i.test(message)))
    if (guard)
      throw Object.assign(new Error('Claude publisher blocked by upstream workflow validation. No fallback token will be used.'), {
        code: 'WORKFLOW_VALIDATION_FAILED',
      })
    throw new Error(`Claude publisher token exchange failed (HTTP ${response.status}).`)
  }
  const body = await readJson(response)
  const token = body.token || body.app_token
  if (!validToken(token)) throw new Error('Claude publisher exchange returned no valid app token.')
  mask(token)
  return token
}

export async function revokeToken(env, request = fetch) {
  if (!env.CLAUDE_PUBLISH_TOKEN) return
  const response = await request(`${env.GITHUB_API_URL || 'https://api.github.com'}/installation/token`, {
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${env.CLAUDE_PUBLISH_TOKEN}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
    signal: timeout(),
  })
  if (!response.ok && response.status !== 401) {
    throw new Error(`Claude publisher token revocation failed (HTTP ${response.status}).`)
  }
}

export function writeGuardSummary(error, summaryPath) {
  if (error.code !== 'WORKFLOW_VALIDATION_FAILED' || !summaryPath) return
  appendFileSync(
    summaryPath,
    '\n### Claude review blocked by workflow validation\n\n' +
      'The Claude token exchange explicitly refused this workflow. No review was run ' +
      'and no alternative publishing identity was used. This is an authentication guard, ' +
      'not a finding about the PR code. Check the workflow against the default branch ' +
      'and retry from an allowed workflow context after resolving the refusal.\n',
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    if (process.argv[2] === 'revoke') {
      await revokeToken(process.env)
    } else if (process.argv[2] === 'acquire') {
      if (!process.env.GITHUB_OUTPUT) throw new Error('GITHUB_OUTPUT is required.')
      const token = await acquireToken(process.env, fetch, (value) => console.log(`::add-mask::${value}`))
      try {
        appendFileSync(process.env.GITHUB_OUTPUT, `token=${token}\n`)
      } catch {
        await revokeToken({ ...process.env, CLAUDE_PUBLISH_TOKEN: token })
        throw new Error('Could not register publisher token for cleanup.')
      }
    } else {
      throw new Error('Use acquire or revoke.')
    }
  } catch (error) {
    try {
      writeGuardSummary(error, process.env.GITHUB_STEP_SUMMARY)
    } catch {
      console.error('::warning::Could not write the workflow-validation explanation to the step summary.')
    }
    console.error(`::error::${error.message}`)
    process.exitCode = 1
  }
}
