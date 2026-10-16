import numpy as np

F = np.float32

# =========================================================
# Geometria dell'anello (identica a layout.csl)
# =========================================================

def ring_k(px, py, side):
    if py % 2 == 0:
        return py * side + px
    return py * side + (side - 1 - px)

def pe_order(side):
    """Restituisce la lista dei (px,py) ordinati per k crescente."""
    out = [None] * (side * side)
    for py in range(side):
        for px in range(side):
            out[ring_k(px, py, side)] = (px, py)
    return out

class NBodyFP32:
    """Replica bit-a-bit del kernel CS-3 (versione con accumulatori separati).

    Ordine di riduzione per il PE k:
        self  : j = 0 .. N_LOCAL-1   (incluso j == i)
        fwd   : chunk k-1, k-2, ..., 0
        bwd   : chunk k+1, k+2, ..., ring_len-1
        totale: (self + fwd) + bwd
    """

    def __init__(self, side, n_local, n_steps, dt_num, dt_den,
                 soft_eps_sq=1e-4, G=1.0):
        self.side = side
        self.n_local = n_local
        self.n_steps = n_steps
        self.ring_len = side * side

        # DT e HALF_DT arrotondati a f32 una volta sola, come in constants.csl
        self.DT = F(F(dt_num) / F(dt_den))
        self.HALF_DT = F(self.DT * F(0.5))
        self.EPS2 = F(soft_eps_sq)
        self.G = F(G)

    # ---- kernel di coppia: una sorgente j contro tutte le N_LOCAL locali ----
    def _pair_update(self, xi, yi, zi, mi, xj, yj, zj, mj, fx, fy, fz):
        # dx = xj - xi  (fmov + fmac fused: xj - 1.0*xi, un solo arrotondamento)
        dx = F(xj - xi)
        dy = F(yj - yi)
        dz = F(zj - zi)

        # r2 costruito con mul e add SEPARATI (niente FMA)
        r2 = np.full_like(xi, self.EPS2)
        r2 = F(r2 + F(dx * dx))
        r2 = F(r2 + F(dy * dy))
        r2 = F(r2 + F(dz * dz))

        r = np.sqrt(r2, dtype=F)
        r3 = F(r * r2)
        inv_r3 = F(F(1.0) / r3)

        f_scal = F(mi * inv_r3)
        G_mj = F(self.G * F(mj))        # scalare, precomputato
        f_scal = F(f_scal * G_mj)

        # accumulo: mul in tmp, poi add separato
        fx += F(f_scal * dx); fx[:] = F(fx)
        fy += F(f_scal * dy); fy[:] = F(fy)
        fz += F(f_scal * dz); fz[:] = F(fz)

    def _accumulate_chunk(self, st, k, src_x, src_y, src_z, src_m, fx, fy, fz):
        """Accumula il contributo di un chunk (N_LOCAL sorgenti) sul PE k."""
        xi, yi, zi = st['x'][k], st['y'][k], st['z'][k]
        mi = st['m'][k]
        for j in range(self.n_local):
            self._pair_update(xi, yi, zi, mi,
                              src_x[j], src_y[j], src_z[j], src_m[j],
                              fx, fy, fz)

    def _compute_forces(self, st):
        """Forze su tutti i PE, con l'ordine di riduzione del kernel."""
        R, NL = self.ring_len, self.n_local
        fx_tot = np.zeros((R, NL), dtype=F)
        fy_tot = np.zeros((R, NL), dtype=F)
        fz_tot = np.zeros((R, NL), dtype=F)

        for k in range(R):
            # --- self ---
            fx_s = np.zeros(NL, dtype=F)
            fy_s = np.zeros(NL, dtype=F)
            fz_s = np.zeros(NL, dtype=F)
            self._accumulate_chunk(st, k, st['x'][k], st['y'][k],
                                   st['z'][k], st['m'][k], fx_s, fy_s, fz_s)

            # --- fwd: predecessori in ordine k-1, k-2, ..., 0 ---
            fx_f = np.zeros(NL, dtype=F)
            fy_f = np.zeros(NL, dtype=F)
            fz_f = np.zeros(NL, dtype=F)
            for src in range(k - 1, -1, -1):
                self._accumulate_chunk(st, k, st['x'][src], st['y'][src],
                                       st['z'][src], st['m'][src],
                                       fx_f, fy_f, fz_f)

            # --- bwd: successori in ordine k+1, k+2, ..., R-1 ---
            fx_b = np.zeros(NL, dtype=F)
            fy_b = np.zeros(NL, dtype=F)
            fz_b = np.zeros(NL, dtype=F)
            for src in range(k + 1, R):
                self._accumulate_chunk(st, k, st['x'][src], st['y'][src],
                                       st['z'][src], st['m'][src],
                                       fx_b, fy_b, fz_b)

            # --- combine_forces_soa: (self + fwd) + bwd ---
            fx_tot[k] = F(F(fx_s + fx_f) + fx_b)
            fy_tot[k] = F(F(fy_s + fy_f) + fy_b)
            fz_tot[k] = F(F(fz_s + fz_f) + fz_b)

        return fx_tot, fy_tot, fz_tot

    def run(self, particles_by_pe):
        """particles_by_pe: array (ring_len, n_local, 7) = [x,y,z,vx,vy,vz,m]"""
        p = np.ascontiguousarray(particles_by_pe, dtype=F)
        st = {
            'x': p[:, :, 0].copy(), 'y': p[:, :, 1].copy(), 'z': p[:, :, 2].copy(),
            'vx': p[:, :, 3].copy(), 'vy': p[:, :, 4].copy(), 'vz': p[:, :, 5].copy(),
            'm': p[:, :, 6].copy(),
        }
        inv_m = F(F(1.0) / st['m'])

        step = 0
        while True:
            fx, fy, fz = self._compute_forces(st)
            ax = F(fx * inv_m)
            ay = F(fy * inv_m)
            az = F(fz * inv_m)

            if step == self.n_steps:
                st['vx'] = F(st['vx'] + F(ax * self.HALF_DT))
                st['vy'] = F(st['vy'] + F(ay * self.HALF_DT))
                st['vz'] = F(st['vz'] + F(az * self.HALF_DT))
                break

            if step == 0:
                st['vx'] = F(st['vx'] + F(ax * (-self.HALF_DT)))
                st['vy'] = F(st['vy'] + F(ay * (-self.HALF_DT)))
                st['vz'] = F(st['vz'] + F(az * (-self.HALF_DT)))

            st['vx'] = F(st['vx'] + F(ax * self.DT))
            st['vy'] = F(st['vy'] + F(ay * self.DT))
            st['vz'] = F(st['vz'] + F(az * self.DT))

            st['x'] = F(st['x'] + F(st['vx'] * self.DT))
            st['y'] = F(st['y'] + F(st['vy'] * self.DT))
            st['z'] = F(st['z'] + F(st['vz'] * self.DT))

            step += 1

        out = np.empty_like(p)
        out[:, :, 0] = st['x']; out[:, :, 1] = st['y']; out[:, :, 2] = st['z']
        out[:, :, 3] = st['vx']; out[:, :, 4] = st['vy']; out[:, :, 5] = st['vz']
        out[:, :, 6] = st['m']
        forces = np.stack([fx, fy, fz], axis=-1)
        return out, forces

side, n_local, n_steps = 4, 8, 100
dt_num, dt_den = 1, 1000

inp = np.load("input.npy")           # shape attesa (ring_len, n_local, 7)
ref = np.load("output_cs3.npy")

sim = NBodyFP32(side, n_local, n_steps, dt_num, dt_den)
out, forces = sim.run(inp)

exact = np.array_equal(out.view(np.uint32), ref.view(np.uint32))
print("bit-esatto:", exact)
if not exact:
    ulp = np.abs(out.view(np.int32) - ref.view(np.int32))
    print("max ULP:", ulp.max(), "| frazione non esatta:", (ulp != 0).mean())
