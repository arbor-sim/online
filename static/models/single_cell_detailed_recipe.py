import arbor
import pandas
import sys
from arbor import density
from plotly.subplots import make_subplots
import plotly.express as px
import arbor_playground

# (1) Read the morphology from an SWC file.

# WARNING! Loading a non-existing file will crash pyodide

filename = 'single_cell_detailed.swc'
morph = arbor.load_swc_arbor(filename)

print('SWC contents:')
with open(filename) as f:
    print(f.read())

# (2) Create and populate the label dictionary.

labels = arbor.label_dict(
    {
        # Regions:
        # Add a label for a region that includes the whole morphology
        "all": "(all)",
        # Add a label for the parts of the morphology with radius greater than 1.5 μm.
        "gt_1.5": '(radius-ge (region "all") 1.5)',
        # Join regions "apic" and "gt_1.5"
        "custom": '(join (region "apic") (region "gt_1.5"))',
        # Locsets:
        # Add a labels for the root of the morphology and all the terminal points
        "root": "(root)",
        "terminal": "(terminal)",
        # Add a label for the terminal locations in the "custom" region:
        "custom_terminal": '(restrict (locset "terminal") (region "custom"))',
        # Add a label for the terminal locations in the "axon" region:
        "axon_terminal": '(restrict (locset "terminal") (region "axon"))',
    }
).add_swc_tags()  # Add SWC pre-defined regions

# (3) Create and populate the decor.

decor = (
    arbor.decor()
    # Set the default properties of the cell (this overrides the model defaults).
    .set_property(Vm=-55)
    .set_ion("na", int_con=10, ext_con=140, rev_pot=50, method="nernst/na")
    .set_ion("k", int_con=54.4, ext_con=2.5, rev_pot=-77)
    # Override the cell defaults.
    .paint('"custom"', tempK=270)
    .paint('"soma"', Vm=-50)
    # Paint density mechanisms.
    .paint('"all"', density("pas"))
    .paint('"custom"', density("hh"))
    .paint('"dend"', density("Ih", {"gbar": 0.001}))
    # Place stimuli and detectors.
    .place('"root"', arbor.iclamp(10, 1, current=2), "iclamp0")
    .place('"root"', arbor.iclamp(30, 1, current=2), "iclamp1")
    .place('"root"', arbor.iclamp(50, 1, current=2), "iclamp2")
    .place('"axon_terminal"', arbor.threshold_detector(-10), "detector")
    # Set discretisation: Soma as one CV, 1um everywhere else
    .discretization('(replace (single (region "soma")) (max-extent 1.0))')
)


# (4) Create the cell.

cell = arbor.cable_cell(morph, decor, labels)


# (5) Create a class that inherits from arbor.recipe
class single_recipe(arbor.recipe):

    # (5.1) Define the class constructor
    def __init__(self):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)

        self.the_props = arbor.cable_global_properties()
        self.the_props.set_property(Vm=-65, tempK=300, rL=35.4, cm=0.01)
        self.the_props.set_ion(
            ion="na", int_con=10, ext_con=140, rev_pot=50, method="nernst/na"
        )
        self.the_props.set_ion(ion="k", int_con=54.4, ext_con=2.5, rev_pot=-77)
        self.the_props.set_ion(ion="ca", int_con=5e-5, ext_con=2, rev_pot=132.5)
        self.the_props.catalogue.extend(arbor.allen_catalogue(), "")

    # (5.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (5.3) Override the cell_kind method
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (5.4) Override the cell_description method
    def cell_description(self, gid):
        return cell

    # (5.5) Override the probes method
    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('"custom_terminal"')]

    # (5.6) Override the global_properties method
    def global_properties(self, gid):
        return self.the_props


# Instantiate recipe
recipe = single_recipe()

# (6) Create a simulation
sim = arbor.simulation(recipe) # this here is out of range?

# Instruct the simulation to record the spikes and sample the probe
sim.record(arbor.spike_recording.all)

probeset_id = arbor.cell_member(0, 0)
handle = sim.sample(probeset_id, arbor.regular_schedule(0.02))

# (7) Run the simulation
sim.run(tfinal=100, dt=0.025)

# (8) Print or display the results
spikes = sim.spikes()
print(len(spikes), "spikes recorded:")
for s in spikes:
    print(s)

data = []
meta = []
for d, m in sim.samples(handle):
    data.append(d)
    meta.append(m)
# Plot voltages
df_list = []
for i in range(len(data)):
    df_list.append(
        pandas.DataFrame(
            {
                "t/ms": data[i][:, 0],
                "U/mV": data[i][:, 1],
                "Location": str(meta[i]),
                "Variable": "voltage",
            }
        )
    )
df = pandas.concat(df_list, ignore_index=True)
fig_trace = px.line(df, x="t/ms", y="U/mV", color="Location")

# Plot morphology
segments = [(branch, i, seg)
                for branch in range(morph.num_branches)
                for i, seg in enumerate(morph.branch_segments(branch))]
seg_data = []
for branch, i, segment in segments:
    assert segment.prox.z == 0 # example neuron is 2d
    name = f'branch{branch}/seg{i}'
    seg_data.append(dict(
        segment=name,
        branch=branch,
        x=segment.prox.x,
        y=segment.prox.y
        ))
    # add midpoint for labels
    seg_data.append(dict(
        segment=name,
        branch=branch,
        x=(segment.prox.x+segment.dist.x)/2,
        y=(segment.prox.y+segment.dist.y)/2
        ))
    seg_data.append(dict(
        segment=name,
        branch=branch,
        x=segment.dist.x,
        y=segment.dist.y
        ))
seg_data = pandas.DataFrame(seg_data)
fig_morph = px.line(seg_data, x='x', y='y', color='branch', hover_name='segment')

# Combine plots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Voltage Trace", "Morphology"))
for trace in range(len(fig_trace["data"])):
    fig.append_trace(fig_trace["data"][trace], row=1, col=1)
for trace in range(len(fig_morph["data"])):
    fig.append_trace(fig_morph["data"][trace], row=1, col=2)
fig_html = fig.to_html(include_plotlyjs=False, full_html=False)
arbor_playground.render_html(fig_html)
