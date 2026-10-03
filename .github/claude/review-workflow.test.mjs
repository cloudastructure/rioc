import assert from 'node:assert/strict'
import { readFileSync, mkdtempSync, mkdirSync, writeFileSync, rmSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import test from 'node:test'

const workflow = readFileSync(new URL('../workflows/claude.yml', import.meta.url), 'utf8')
const auto = workflow.split('  auto-review:')[1].split('  review-status:')[0]
const mention = workflow.split('  claude:')[1]

test('bootstrap skips branch copies until the required runtime exists on main', () => {
  const step = auto.split('      - name: Check the trusted bot files are complete\n')[1].split('\n      # Separate')[0]
  const script = step.split('        run: |\n')[1].split('\n').map((line) => line.replace(/^          /, '')).join('\n')
  const dir = mkdtempSync(join(tmpdir(), 'review-bootstrap-'))
  try {
    const git = (...args) => execFileSync('git', args, { cwd: dir, stdio: 'pipe' })
    git('init', '-b', 'main')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '--allow-empty', '-m', 'Base')
    git('update-ref', 'refs/remotes/origin/main', 'HEAD')
    mkdirSync(join(dir, '.github/claude'), { recursive: true })
    for (const name of readdirSync(new URL('.', import.meta.url))) {
      if (/\.(sh|json|md|mjs)$/.test(name)) writeFileSync(join(dir, '.github/claude', name), '')
    }
    const run = () => {
      const output = join(dir, 'output')
      writeFileSync(output, '')
      execFileSync('bash', ['-e', '-c', script], { cwd: dir, env: { ...process.env, GITHUB_OUTPUT: output, GITHUB_STEP_SUMMARY: join(dir, 'summary') }, stdio: 'pipe' })
      return readFileSync(output, 'utf8')
    }
    assert.match(run(), /missing=true/)
    assert.match(readFileSync(join(dir, 'summary'), 'utf8'), /cannot run until these trusted files exist on main/)
    git('add', '.github')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-m', 'Add runtime')
    git('update-ref', 'refs/remotes/origin/main', 'HEAD')
    assert.match(run(), /missing=false/)
    rmSync(join(dir, '.github/claude/publish-review.sh'))
    assert.match(run(), /missing=true/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('every automatic model pass allows only the trusted fix bot', () => {
  const passes = auto.split(/      - name: /).filter((step) => step.includes('uses: anthropics/claude-code-action@'))
  assert.equal(passes.length, 3, 'review, verification, and recomposition all validate the triggering actor')
  for (const pass of passes) {
    assert.match(pass, /allowed_bots: "claude\[bot\]"/)
    assert.equal((pass.match(/allowed_bots:/g) || []).length, 1)
  }
  assert.doesNotMatch(mention, /allowed_bots:/)
})

test('pushes do not re-review and bot comments cannot request follow-ups', () => {
  const triggers = workflow.split('  pull_request:\n')[1].split('  workflow_dispatch:')[0]
  assert.match(triggers, /types: \[opened, ready_for_review\]/)
  assert.match(mention, /github\.event\.sender\.type != 'Bot' &&/)
  assert.match(auto, /github\.event\.pull_request\.head\.repo\.full_name == github\.repository &&/)
  assert.match(mention, /contains\(fromJSON\('\["OWNER","MEMBER","COLLABORATOR"\]'\), github\.event\.comment\.author_association \|\| github\.event\.review\.author_association \|\| github\.event\.issue\.author_association\)/)
  assert.match(mention, /contains\(github\.event\.comment\.body, '@claude'\)/)
})

test('both progress reporters receive stage outcomes and execution evidence', () => {
  for (const [job, model] of [
    [auto, 'claude_review'],
    [mention, 'claude_run'],
  ]) {
    const progress = job.split('      - name: Update progress indicator\n')[1].split('\n      - name:')[0]
    for (const input of [
      'MODEL_OUTCOME: ${{ steps.' + model + '.outcome }}',
      'MODEL_EXECUTION_FILE: ${{ steps.' + model + '.outputs.execution_file }}',
      'CAPTURE_OUTCOME: ${{ steps.capture_review.outcome }}',
      'PUBLISH_OUTCOME: ${{ steps.publish_review.outcome }}',
    ])
      assert.ok(progress.includes(input), input)
  }
  assert.match(auto, /failure_summary: \$\{\{ steps\.update_progress\.outputs\.failure_summary \}\}/)
  assert.match(workflow, /FAILURE_SUMMARY: \$\{\{ needs\.auto-review\.outputs\.failure_summary \}\}/)
  assert.match(workflow, /-f conclusion='neutral'/)
  assert.doesNotMatch(workflow, /when the provider is healthy/)
})
