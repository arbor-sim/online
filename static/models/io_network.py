import arbor
import random
import pandas as pd
import plotly.express as px
import arbor_playground

io_catalogue = arbor.load_catalogue('io-catalogue.so')

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
        return random.gauss(x, x/20)
    return (arbor.decor()
        .paint('"soma"', arbor.density("hh"))
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

class NetworkIO(arbor.recipe):
    def __init__(self, ncells):
        super().__init__()
        self.ncells = ncells
        self.props = arbor.neuron_cable_properties()
        self.props.catalogue.extend(io_catalogue, '')
        self.cluster_size = 4
        self.dend_count = 10
        self.bridge_fraction = 0.1

    def cell_description(self, gid):
        tree = arbor.segment_tree()
        soma = tree.append(arbor.mnpos, arbor.mpoint(-12, 0, 0, 12), arbor.mpoint(0, 0, 0, 12), tag=1)
        tree.append(soma, arbor.mpoint(-random.randrange(-50, -40), 0, 0, 2), arbor.mpoint(-12, 0, 0, 2), tag=2) # axon
        labels_dict = dict(LABELS_DEFAULT)
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
        # curly
        is_bridge = random.random() < self.bridge_fraction
        for _ in range(random.randrange(5, 10)):
            i = random.randrange(0, self.dend_count)
            j = random.randrange(0, self.dend_count)
            if is_bridge:
                other = random.randrange(0, self.num_cells() - 1)
                if other >= gid:
                    other += 1
            else:
                other = gid // self.cluster_size * self.cluster_size + random.randrange(0, self.cluster_size)
            if not (0 <= other < self.num_cells()):
                continue
            print(gid, '->', other)
            conns.append(arbor.gap_junction_connection((other, f'"gj{i}"'), f'"gj{j}"', 0.05))
        return conns
    def num_cells(self): return self.ncells
    def cell_kind(self, gid): return arbor.cell_kind.cable
    def probes(self, gid): return [arbor.cable_probe_membrane_voltage('"root"')]
    def global_properties(self, kind): return self.props

recipe = NetworkIO(ncells=16)
sim = arbor.simulation(recipe)
handles = [sim.sample((gid, 0), arbor.regular_schedule(1)) for gid in range(recipe.num_cells())]
sim.run(tfinal=1000, dt=0.1)

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
