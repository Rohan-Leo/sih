/**
 * Unscented Kalman Filter — TypeScript port of ml/clew_ml/ukf.py.
 *
 * State x = [E, N, ψ, v, b_ω, b_a]: local east/north (m), heading (rad,
 * clockwise from north), forward speed (m/s), gyro yaw bias, accel bias.
 * Non-holonomic unicycle motion: the vehicle only moves along its heading.
 */
const N = 6
type Vec = number[]
type Mat = number[][]

const wrap = (a: number) => ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI

function cholesky(A: Mat): Mat | null {
  const L: Mat = A.map(() => new Array(N).fill(0))
  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      if (i === j) {
        if (s <= 0) return null
        L[i][i] = Math.sqrt(s)
      } else L[i][j] = s / L[j][j]
    }
  }
  return L
}

function inv(S: Mat): Mat {
  if (S.length === 1) return [[1 / S[0][0]]]
  const [[a, b], [c, d]] = S
  const det = a * d - b * c
  return [
    [d / det, -b / det],
    [-c / det, a / det],
  ]
}

export class UKF {
  x: Vec = new Array(N).fill(0)
  P: Mat = eye(1)
  private wm: number[]
  private wc: number[]
  private c: number
  private q = [0.05, 0.05, 0.02, 1.0, 2e-4, 5e-3].map((v) => v * v)

  constructor(alpha = 0.5, beta = 2, kappa = 0) {
    const lam = alpha * alpha * (N + kappa) - N
    this.c = N + lam
    this.wm = new Array(2 * N + 1).fill(1 / (2 * this.c))
    this.wc = [...this.wm]
    this.wm[0] = lam / this.c
    this.wc[0] = lam / this.c + (1 - alpha * alpha + beta)
  }

  init(E: number, Nn: number, psi: number, v: number) {
    this.x = [E, Nn, psi, v, 0, 0]
    this.P = diag([5, 5, (10 * Math.PI) / 180, 1, 0.02, 0.2].map((s) => s * s))
  }

  get sigmaPos(): number {
    // largest eigenvalue of the 2×2 position covariance
    const a = this.P[0][0]
    const b = this.P[0][1]
    const d = this.P[1][1]
    const tr = a + d
    const det = a * d - b * b
    return Math.sqrt(Math.max(0, tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det))))
  }

  private sigmas(): Vec[] {
    let L = cholesky(this.P.map((r) => r.map((v) => v * this.c)))
    if (!L) {
      this.P = this.P.map((r, i) => r.map((v, j) => 0.5 * (v + this.P[j][i]) + (i === j ? 1e-6 : 0)))
      L = cholesky(this.P.map((r) => r.map((v) => v * this.c))) ?? eye(Math.sqrt(this.c) * 1e-3)
    }
    const X: Vec[] = [this.x.slice()]
    for (let j = 0; j < N; j++) X.push(this.x.map((v, i) => v + L![i][j]))
    for (let j = 0; j < N; j++) X.push(this.x.map((v, i) => v - L![i][j]))
    return X
  }

  private meanX(X: Vec[]): Vec {
    const m = new Array(N).fill(0)
    let s = 0
    let c = 0
    X.forEach((p, k) => {
      for (let i = 0; i < N; i++) m[i] += this.wm[k] * p[i]
      s += this.wm[k] * Math.sin(p[2])
      c += this.wm[k] * Math.cos(p[2])
    })
    m[2] = Math.atan2(s, c)
    return m
  }

  predict(omega: number, aLong: number, dt: number) {
    const Y = this.sigmas().map((p) => {
      const psi = p[2] + (omega - p[4]) * dt
      const v = Math.max(0, p[3] + (aLong - p[5]) * dt)
      const vm = 0.5 * (p[3] + v)
      const pm = p[2] + 0.5 * (psi - p[2])
      return [p[0] + vm * Math.sin(pm) * dt, p[1] + vm * Math.cos(pm) * dt, wrap(psi), v, p[4], p[5]]
    })
    this.x = this.meanX(Y)
    const P = zeros()
    Y.forEach((p, k) => {
      const d = p.map((v, i) => (i === 2 ? wrap(v - this.x[2]) : v - this.x[i]))
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) P[i][j] += this.wc[k] * d[i] * d[j]
    })
    for (let i = 0; i < N; i++) P[i][i] += this.q[i] * dt
    this.P = P
  }

  /** Generic update; h maps a sigma point to a measurement vector. */
  update(z: Vec, h: (p: Vec) => Vec, R: Mat, angleRows: number[] = [], gate?: number): boolean {
    const X = this.sigmas()
    const Z = X.map(h)
    const m = z.length
    const zm = new Array(m).fill(0)
    Z.forEach((q, k) => q.forEach((v, i) => (zm[i] += this.wm[k] * v)))
    for (const r of angleRows) {
      let s = 0
      let c = 0
      Z.forEach((q, k) => {
        s += this.wm[k] * Math.sin(q[r])
        c += this.wm[k] * Math.cos(q[r])
      })
      zm[r] = Math.atan2(s, c)
    }
    const S: Mat = R.map((r) => r.slice())
    const C: Mat = Array.from({ length: N }, () => new Array(m).fill(0))
    X.forEach((p, k) => {
      const dz = Z[k].map((v, i) => (angleRows.includes(i) ? wrap(v - zm[i]) : v - zm[i]))
      const dx = p.map((v, i) => (i === 2 ? wrap(v - this.x[2]) : v - this.x[i]))
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) S[i][j] += this.wc[k] * dz[i] * dz[j]
      for (let i = 0; i < N; i++) for (let j = 0; j < m; j++) C[i][j] += this.wc[k] * dx[i] * dz[j]
    })
    const y = z.map((v, i) => (angleRows.includes(i) ? wrap(v - zm[i]) : v - zm[i]))
    const Si = inv(S)
    if (gate !== undefined) {
      let d2 = 0
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) d2 += y[i] * Si[i][j] * y[j]
      if (d2 > gate) return false // innovation gating: reject inconsistent measurements
    }
    const K: Mat = C.map((row) => Si[0].map((_, j) => row.reduce((a, c, l) => a + c * Si[l][j], 0)))
    for (let i = 0; i < N; i++) this.x[i] += K[i].reduce((a, k, j) => a + k * y[j], 0)
    this.x[2] = wrap(this.x[2])
    this.x[3] = Math.max(0, this.x[3])
    // P ← P − K S Kᵀ
    const KS = K.map((row) => S[0].map((_, j) => row.reduce((a, k, l) => a + k * S[l][j], 0)))
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) this.P[i][j] -= KS[i].reduce((a, v, l) => a + v * K[j][l], 0)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < i; j++) this.P[i][j] = this.P[j][i] = 0.5 * (this.P[i][j] + this.P[j][i])
    return true
  }

  gnss(E: number, Nn: number, acc: number, speed: number | null, courseDeg: number | null) {
    const r = Math.max(acc, 2) ** 2
    this.update([E, Nn], (p) => [p[0], p[1]], [[r, 0], [0, r]], [], 25)
    if (speed !== null) this.update([speed], (p) => [p[3]], [[0.09]])
    if (courseDeg !== null && speed !== null && speed > 3) {
      this.update([(courseDeg * Math.PI) / 180], (p) => [p[2]], [[(3 * Math.PI / 180) ** 2]], [0])
    }
  }

  zupt() {
    this.update([0], (p) => [p[3]], [[0.0025]])
  }

  speedObs(v: number, variance: number) {
    this.update([v], (p) => [p[3]], [[variance]])
  }

  position(E: number, Nn: number, sigma: number) {
    this.update([E, Nn], (p) => [p[0], p[1]], [[sigma * sigma, 0], [0, sigma * sigma]])
  }
}

function zeros(): Mat {
  return Array.from({ length: N }, () => new Array(N).fill(0))
}
function eye(s: number): Mat {
  return diag(new Array(N).fill(s))
}
function diag(d: number[]): Mat {
  const M = zeros()
  d.forEach((v, i) => (M[i][i] = v))
  return M
}
