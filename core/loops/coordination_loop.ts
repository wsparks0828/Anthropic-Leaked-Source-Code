/**
 * Loop 6 — Coordination (barrier) Loop.
 *
 * Fan-out a decision to N independent verifier functions, then join at a barrier
 * with a bounded wait. Fail-closed: if ANY verifier rejects, errors, or the
 * barrier times out, the joined verdict is 'fail'. No partial-quorum accept.
 *
 * INVARIANTS:
 *   I1: join verdict 'pass' ⇒ ALL verifiers returned 'approve' within the deadline
 *   I2: any timeout/error/reject ⇒ verdict 'fail' (never 'pass', never 'warn'-upgraded)
 *   I3: the number of votes counted == number of verifiers dispatched (no dropped votes)
 */

export type VerifierVote = 'approve' | 'reject'
export type BarrierVerdict = 'pass' | 'fail'

export interface VerifierFn {
  id: string
  run: () => Promise<VerifierVote>
}

export interface BarrierOutcome {
  verdict: BarrierVerdict
  votes: Array<{id: string; vote: VerifierVote | 'timeout' | 'error'}>
  dispatched: number
  failClosedReason?: string
}

export class CoordinationLoop {
  readonly id = 'coordination' as const

  /**
   * Gather verifier votes at a barrier with a bounded deadline.
   * Pure async; deterministic given verifier fns + timeout.
   */
  async gather(verifiers: VerifierFn[], timeoutMs = 1000): Promise<BarrierOutcome> {
    const dispatched = verifiers.length
    const votes: Array<{id: string; vote: VerifierVote | 'timeout' | 'error'}> = []

    const settled = await Promise.all(
      verifiers.map(async (v) => {
        try {
          const vote = await this.withDeadline(v.run(), timeoutMs)
          return {id: v.id, vote}
        } catch (e) {
          // timeout or thrown error → fail-closed contribution
          const isTimeout = e instanceof Error && e.message === '__barrier_timeout__'
          return {id: v.id, vote: (isTimeout ? 'timeout' : 'error') as 'timeout' | 'error'}
        }
      }),
    )

    for (const s of settled) votes.push(s)

    // I3: every dispatched verifier produced exactly one vote slot.
    if (votes.length !== dispatched) {
      return {verdict: 'fail', votes, dispatched, failClosedReason: 'vote count mismatch'}
    }

    // I1/I2: pass iff ALL approve; anything else fails closed.
    const allApprove = votes.every((x) => x.vote === 'approve')
    if (allApprove) {
      return {verdict: 'pass', votes, dispatched}
    }

    const offender = votes.find((x) => x.vote !== 'approve')
    return {
      verdict: 'fail',
      votes,
      dispatched,
      failClosedReason: `verifier ${offender?.id} → ${offender?.vote}`,
    }
  }

  private withDeadline<T>(p: Promise<T>, ms: number): Promise<T> {
    const timeout = new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error('__barrier_timeout__')), ms),
    )
    return Promise.race([p, timeout])
  }

  assertInvariant(): void {
    /* stateless; invariants enforced inline per gather() */
  }

  reset(): void {
    /* stateless */
  }
}

export const globalCoordinationLoop = new CoordinationLoop()
