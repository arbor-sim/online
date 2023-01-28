#!/usr/bin/env python3

import os, sys
import numpy as np
import subprocess as sp
from pathlib import Path
from time import perf_counter as pc
import arbor as A
import arbor_playground

catalogue_path = os.path.join(A.__path__[0], 'l5pc-catalogue.so')
morphology_path = 'l5pc.nml'
acc_path = f'l5pc.acc'

FIGURE = '5a'
assert FIGURE in {'4b', '4a', '5a'}
dt = 0.025
tstop = 3000 if FIGURE == '4b' else 600

class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.props = A.neuron_cable_properties()
        self.props.catalogue.extend(A.load_catalogue(catalogue_path), 'local_')
        self.cell_to_morph = {'L5PC': 'morphology_L5PC', }
        self.gid_to_cell = ['L5PC', ]
        if FIGURE == '4b':
            self.i_clamps = {'Input_0': (699.999988079071, 2000.0, 0.7929999989997327), }
        elif FIGURE == '4a':
            self.i_clamps = {'Input_0': (295.0, 5.0, 1.9), }
        elif FIGURE == '5a':
            self.i_clamps = {'Input_0': [(250 + i*1000/120, 5, 1.99) for i in range(5)]}
        self.gid_to_inputs = { 0: [("seg_0_frac_0.5", "Input_0"), ], }
        self.gid_to_synapses = { }
        self.gid_to_detectors = { }
        self.gid_to_connections = { }
        self.gid_to_labels = { 0: [(0, 0.5), ], }

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, gid):
        cid = self.gid_to_cell[gid]
        mrf = self.cell_to_morph[cid]
        nml = A.neuroml(morphology_path).morphology(mrf, allow_spherical_root=True)
        lbl = A.label_dict()
        lbl.append(nml.segments())
        lbl.append(nml.named_segments())
        lbl.append(nml.groups())
        lbl['all'] = '(all)'
        if gid in self.gid_to_labels:
            for seg, frac in self.gid_to_labels[gid]:
                lbl[f'seg_{seg}_frac_{frac}'] = f'(on-components {frac} (segment {seg}))'
        dec = A.load_component(acc_path).component


        #  forsec all {
        #    nseg = 1 + 2*int(L/40)
        #    nSec = nSec + 1
        #  }
        dec.discretization(A.cv_policy_max_extent(20))

        if gid in self.gid_to_inputs:
            for tag, inp in self.gid_to_inputs[gid]:
                x = self.i_clamps[inp]
                if not isinstance(x, list):
                    x = [x]
                for lag, dur, amp in x:
                    dec.place(f'"{tag}"', A.iclamp(lag, dur, amp), f'ic_{inp}@{tag}')
        if gid in self.gid_to_synapses:
            for tag, syn in self.gid_to_synapses[gid]:
                dec.place(f'"{tag}"', A.synapse(syn), f'syn_{syn}@{tag}')
        if gid in self.gid_to_detectors:
            for tag in self.gid_to_detectors[gid]:
                dec.place(f'"{tag}"', A.spike_detector(-40), f'sd@{tag}') # -40 is a phony value!!!
        globals().update(locals())
        self.mrf = nml.morphology
        return A.cable_cell(nml.morphology, dec, lbl)

    def probes(self, _):
        # Example: probe center of the root (likely the soma)
        return [
                A.cable_probe_membrane_voltage('(location 0 0.5)'),
                A.cable_probe_membrane_voltage_cell()
                ]

    def global_properties(self, kind):
        return self.props

    def connections_on(self, gid):
        res = []
        if gid in self.gid_to_connections:
            for src, dec, syn, loc, w, d in self.gid_to_connections[gid]:
                conn = A.connection((src, f'sd@{dec}'), f'syn_{syn}@{loc}', w, d)
                res.append(conn)
        return res


ctx = A.context(threads=1)
meter_manager = A.meter_manager()
meter_manager.start(ctx)
mdl = recipe()
meter_manager.checkpoint('recipe-create', ctx)
ddc = A.partition_load_balance(mdl, ctx)
meter_manager.checkpoint('load-balance', ctx)
sim = A.simulation(mdl, ctx, ddc)
meter_manager.checkpoint('simulation-init', ctx)

hdl = sim.sample((0, 1), A.regular_schedule(.1))

print(f'Running simulation for {tstop}ms...')

t0 = pc()
sim.run(tstop, dt)
t1 = pc()
print(f'Simulation done, took: {t1-t0:.3f}s')
meter_manager.checkpoint('simulation-run', ctx)

(data, meta), = sim.samples(hdl)
t = data[:,0]
vs = data[:,1:] # (time, cv)

#cc = mdl.cell_description(0)

def point(p):
    return np.array([p.x, p.y, p.z])
def toloc(at):
    segs = mdl.mrf.branch_segments(at.branch)
    ls = np.array([np.linalg.norm(point(s.dist) - point(s.prox)) for s in segs])
    ls = ls / ls.sum()
    ls = ls.cumsum()
    idx = (((at.prox + at.dist) - ls *2)**2).argmin()
    return 0.5*point(segs[idx].prox) + 0.5*point(segs[idx].dist)
locs = np.array([toloc(b) for b in meta])


import io
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 12))
Vmin, Vptp = vs.min(), vs.ptp()
print(vs.max(1).shape)
cmap = plt.get_cmap('RdYlBu')
for i in range(mdl.mrf.num_branches):
    segs = mdl.mrf.branch_segments(i)
    for seg in segs:
        a = point(seg.prox)
        b = point(seg.dist)
        segp = np.vstack([a, b])
        x = np.linalg.norm(locs - (0.5 * a + 0.5 * b), axis=1)
        v = vs[:,x.argmin()]
        c = (v.max() - Vmin) / Vptp
        plt.plot(segp[:,0], segp[:,1], color=cmap(c),
                lw=(seg.dist.radius+seg.prox.radius),
                solid_capstyle='round', zorder=segp[:,2].mean()/100)
skip = 2000
end = -2000
seen = []
for i, (loc, v) in enumerate(zip(locs, vs.T)):
    x, y, z = loc
    skipthis = False
    place_x = x + np.sign(x)*100
    for a, b in seen:
        if np.sqrt(0.3*(a - place_x)**2 + (b - y)**2) < 100:
            skipthis = True
            break
    if skipthis:
        continue
    seen.append((place_x, y))
    xoff = -t[skip:end].mean()
    yoff = -v[skip]
    plt.plot(place_x + (xoff + t[skip:end])*1, y + (yoff + v[skip:end])*1, color='black', lw=1)
    plt.scatter(x, y, marker='<' if x > 0 else '>', color='black', s=10)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
tmpfile = io.BytesIO()
plt.savefig(tmpfile, format='svg')
svg = tmpfile.getvalue()
arbor_playground.render_html(svg)
