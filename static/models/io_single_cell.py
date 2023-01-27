import arbor
import pandas as pd
import plotly.express as px
import arbor_playground

tree = arbor.segment_tree()
soma = tree.append(arbor.mnpos, arbor.mpoint(-12, 0, 0, 12), arbor.mpoint(0, 0, 0, 12), tag=1)
axon = tree.append(soma, arbor.mpoint(-40, 0, 0, 2), arbor.mpoint(-12, 0, 0, 2), tag=2)
for _ in range(15):
    dend = tree.append(soma, arbor.mpoint(6, 0, 0, 2), arbor.mpoint(200, 0, 0, 2), tag=3)

labels = arbor.label_dict({
    'soma': '(tag 1)',
    'axon': '(tag 2)',
    'dend': '(tag 3)',
    'all' : '(all)',
    'root': '(root)'
})
decor = (
    arbor.decor()
    .paint('"soma"', arbor.density("hh"))
    .paint('"soma"', arbor.density('na_s', dict(conductance=0.030)))
    .paint('"soma"', arbor.density('kdr',  dict(conductance=0.030, ek=-75)))
    .paint('"soma"', arbor.density('cal',  dict(conductance=0.045)))
    .paint('"dend"', arbor.density('cah',  dict(conductance=0.010)))
    .paint('"dend"', arbor.density('kca',  dict(conductance=0.220, ek=-75)))
    .paint('"dend"', arbor.density('h',    dict(conductance=0.015, eh=-43)))
    .paint('"dend"', arbor.density('cacc', dict(conductance=0.000)))
    .paint('"axon"', arbor.density('na_a', dict(conductance=0.200)))
    .paint('"axon"', arbor.density('k',    dict(conductance=0.200, ek=-75)))
    .paint('"soma"', arbor.density('k',    dict(conductance=0.015, ek=-75)))
    .paint('"all"',  arbor.density('leak', dict(conductance=1.3e-05, eleak=10)))
    .set_property(cm=0.01) # Ohm.cm
    .set_property(Vm=-65.0)
    .paint('"all"', rL=100) # Ohm.cm
    .paint('"all"', ion_name='ca', rev_pot=120)
    .paint('"all"', ion_name='na', rev_pot=55)
    .paint('"all"', ion_name='k', rev_pot=-75)
    .paint('"all"', arbor.density('ca_conc', dict(initialConcentration=3.7152)))
)

cell = arbor.cable_cell(tree, decor, labels)
m = arbor.single_cell_model(cell)
m.properties.catalogue.extend(arbor.load_catalogue('io-catalogue.so'), '')
m.probe('voltage', where='"root"', frequency=1) 
m.run(2000, 0.1)


df = pd.DataFrame({"t/ms": m.traces[0].time, "U/mV": m.traces[0].value})
fig = px.line(df, x='t/ms', y='U/mV')
fig_html = fig.to_html(include_plotlyjs=False, full_html=False)
arbor_playground.render_html(fig_html)
