import os
import arbor
import random
import numpy as np
import pandas as pd
import plotly.express as px
import arbor_playground

random.seed(0)
np.random.seed(0)

catalogue_path = os.path.join(arbor.__path__[0], 'io-catalogue.so')
io_catalogue = arbor.load_catalogue(catalogue_path)

random.seed(0)

LABELS_DEFAULT = {
    'soma': '(tag 1)',
    'axon': '(tag 2)',
    'dend': '(tag 3)',
    'all' : '(all)',
    'root': '(root)'
}

def decor_default():
    def R(x):
        return random.gauss(x, x/5)
    return (arbor.decor()
        .paint('"soma"', arbor.density('hh'))
        .paint('"soma"', arbor.density('na_s', dict(conductance=R(0.030))))
        .paint('"soma"', arbor.density('kdr',  dict(conductance=R(0.030), ek=-75)))
        .paint('"soma"', arbor.density('cal',  dict(conductance=R(0.045))))
        .paint('"dend"', arbor.density('cah',  dict(conductance=R(0.010))))
        .paint('"dend"', arbor.density('kca',  dict(conductance=R(0.220), ek=-75)))
        .paint('"dend"', arbor.density('h',    dict(conductance=R(0.015), eh=-43)))
        .paint('"dend"', arbor.density('cacc', dict(conductance=R(0.000))))
        .paint('"axon"', arbor.density('na_a', dict(conductance=R(0.200))))
        .paint('"axon"', arbor.density('k',    dict(conductance=R(0.200), ek=-75)))
        .paint('"soma"', arbor.density('k',    dict(conductance=R(0.015), ek=-75)))
        .paint('"all"',  arbor.density('leak', dict(conductance=R(1.3e-05), eleak=10)))
        .set_property(cm=0.01) # Ohm.cm
        .set_property(Vm=-R(65.0))
        .paint('"all"', rL=100) # Ohm.cm
        .paint('"all"', ion_name='ca', rev_pot=R(120))
        .paint('"all"', ion_name='na', rev_pot=R(55))
        .paint('"all"', ion_name='k', rev_pot=-R(75))
        .paint('"all"', arbor.density('ca_conc', dict(initialConcentration=3.7152)))
    )


def build_connection_matrix(cluster_coef, ncells, cluster_size):
    Amask = np.kron(np.eye(int(np.ceil(ncells / cluster_size))), np.ones((cluster_size, cluster_size)))[:ncells,:ncells]
    A = np.random.random((ncells, ncells)) * Amask
    A = (A + A.T)/2
    A = A / A.sum()
    B = np.random.random((ncells, ncells))
    B = (B + B.T)/2
    B = B / B.sum()
    P = cluster_coef * A + (1-cluster_coef) * B
    np.fill_diagonal(P, 0)
    p = np.sort(P.flatten())[::-1]
    p = p[min(ncells * 10, ncells*ncells)]
    return P > p

class NetworkIO(arbor.recipe):
    def __init__(self, ncells):
        super().__init__()
        self.ncells = ncells
        self.props = arbor.neuron_cable_properties()
        self.props.catalogue.extend(io_catalogue, '')
        self.dend_count = 10
        A = build_connection_matrix(cluster_coef=0.8, ncells=self.ncells, cluster_size=4)
        self.conns = {}
        for i, j in zip(*np.where(A)):
            self.conns[min(i, j), max(i, j)] = random.randrange(0, self.dend_count), random.randrange(0, self.dend_count)

    def cell_description(self, gid):
        # starfish topology with gap junctions at each dend end
        #    gj0        gj1
        #      \        /
        #      dend   dend
        #           \  /
        #  --axon--- SOMA----dend---gj2
        #            /  \
        #         dend  dend
        #          /       \
        #        gj4       gj3
        tree = arbor.segment_tree()
        soma = tree.append(arbor.mnpos, arbor.mpoint(-12, 0, 0, 12), arbor.mpoint(0, 0, 0, 12), tag=1)
        tree.append(soma, arbor.mpoint(-random.randrange(-50, -40), 0, 0, 2), arbor.mpoint(-12, 0, 0, 2), tag=2) # axon
        labels_dict = LABELS_DEFAULT.copy()
        decor = decor_default()
        for i in range(self.dend_count):
            tree.append(soma, arbor.mpoint(6, 0, 0, 2), arbor.mpoint(random.randint(180, 250), 0, 0, 2), tag=3) # dend
            labels_dict[f'gj{i}'] = '(location 0 1)'
            decor.place(f'"gj{i}"', arbor.junction('cx36'), f'gj{i}')
        labels = arbor.label_dict(labels_dict)
        cell = arbor.cable_cell(tree, decor, labels)
        return cell

    def gap_junctions_on(self, gid):
        conns = []
        for other in range(self.num_cells()):
            if other == gid: continue
            a, b = min(gid, other), max(gid, other)
            if (a, b) in self.conns:
                i, j = self.conns[a, b]
                if b == gid:
                    i, j = j, i
                conns.append(arbor.gap_junction_connection((other, f'gj{i}'), f'gj{j}', 0.005))
        return conns
    def num_cells(self): return self.ncells
    def cell_kind(self, gid): return arbor.cell_kind.cable
    def probes(self, gid): return [arbor.cable_probe_membrane_voltage('"root"')]
    def global_properties(self, kind): return self.props

recipe = NetworkIO(ncells=16)
sim = arbor.simulation(recipe)
handles = [sim.sample((gid, 0), arbor.regular_schedule(1)) for gid in range(recipe.num_cells())]
sim.run(tfinal=2000, dt=0.025)

for handle in handles:
    data, meta = sim.samples(handle)[0]
    df = pd.DataFrame({"t/ms": data[:, 0], "U/mV": data[:, 1]})
    fig = px.line(df, x='t/ms', y='U/mV')
    fig_html = fig.to_html(include_plotlyjs=False, full_html=False)
    arbor_playground.render_html(fig_html)

df_list = []
for gid, handle in enumerate(handles):
    samples, meta = sim.samples(handle)[0]
    df_list.append(pd.DataFrame({"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Cell": f"Neuron {gid}"}))
df = pd.concat(df_list, ignore_index=True)
fig = px.line(df, x="t/ms", y="U/mV", color='Cell')
fig_html = fig.to_html(include_plotlyjs=False, full_html=False)
arbor_playground.render_html(fig_html)
