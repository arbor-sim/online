import arbor
import argparse
import numpy as np
from numpy.random import RandomState

import pandas as pd
import plotly.express as px
import arbor_playground

"""
A Brunel network consists of nexc excitatory LIF neurons and ninh inhibitory
LIF neurons. Each neuron in the network receives in_degree_prop * nexc
excitatory connections chosen randomly, in_degree_prop * ninh inhibitory
connections and next (external) Poisson connections. All the connections have
the same delay. The strenght of excitatory and Poisson connections is given by
parameter weight, whereas the strength of inhibitory connections is
rel_inh_strength * weight. Poisson neurons all spike independently with expected
number of spikes given by parameter poiss_lambda. Because of the refractory
period, the activity is mostly driven by Poisson neurons and recurrent
connections have a small effect.
"""

# Samples m unique values in interval [start, end) - gid.
# We exclude gid because we don't want self-loops.
def sample_subset(gen, gid, start, end, m):
    idx = np.arange(start, end)
    if start <= gid < end:
        idx = np.delete(idx, gid - start)
    gen.shuffle(idx)
    return idx[:m]
    
class brunel_recipe(arbor.recipe):
    def __init__(
        self,
        nexc,
        ninh,
        next,
        in_degree_prop,
        weight,
        delay,
        rel_inh_strength,
        poiss_lambda,
        seed=42,
    ):

        arbor.recipe.__init__(self)

        # Make sure that in_degree_prop in the interval (0, 1]
        if not 0.0 < in_degree_prop <= 1.0:
            print(
                "The proportion of incoming connections should be in the interval (0, 1]."
            )
            quit()

        self.ncells_exc_ = nexc
        self.ncells_inh_ = ninh
        self.delay_ = delay
        self.seed_ = seed

        # Set up the parameters.
        self.weight_exc_ = weight
        self.weight_inh_ = -rel_inh_strength * weight
        self.weight_ext_ = weight
        self.in_degree_exc_ = round(in_degree_prop * nexc)
        self.in_degree_inh_ = round(in_degree_prop * ninh)
        # each cell receives next incoming Poisson sources with mean rate poiss_lambda, which is equivalent
        # to a single Poisson source with mean rate next*poiss_lambda
        self.lambda_ = next * poiss_lambda

    def num_cells(self):
        return self.ncells_exc_ + self.ncells_inh_

    def cell_kind(self, gid):
        return arbor.cell_kind.lif

    def connections_on(self, gid):
        gen = RandomState(gid + self.seed_)
        connections = []
        # Add incoming excitatory connections.
        connections = [
            arbor.connection((i, "src"), "tgt", self.weight_exc_, self.delay_)
            for i in sample_subset(gen, gid, 0, self.ncells_exc_, self.in_degree_exc_)
        ]
        # Add incoming inhibitory connections.
        connections += [
            arbor.connection((i, "src"), "tgt", self.weight_inh_, self.delay_)
            for i in sample_subset(
                gen,
                gid,
                self.ncells_exc_,
                self.ncells_exc_ + self.ncells_inh_,
                self.in_degree_inh_,
            )
        ]

        return connections

    def cell_description(self, gid):
        cell = arbor.lif_cell("src", "tgt")
        cell.tau_m = 10
        cell.V_th = 10
        cell.C_m = 20
        cell.E_L = 0
        cell.V_m = 0
        cell.t_ref = 2
        return cell

    def event_generators(self, gid):
        t0 = 0
        sched = arbor.poisson_schedule(t0, self.lambda_, gid + self.seed_)
        return [arbor.event_generator("tgt", self.weight_ext_, sched)]

context = arbor.context("avail_threads")
meters = arbor.meter_manager()
meters.start(context)

recipe = brunel_recipe(
    # Number of cells in the excitatory population
    nexc=400,
    # Number of cells in the inhibitory population
    ninh=100,
    # Number of incoming Poisson (external) connections per cell
    next=40,
    # Proportion of the connections received per cell
    in_degree_prop=0.05,
    # Weight of excitatory connections
    weight=1.2,
    # Delay of all connections
    delay=0.1,
    # Relative strength of inhibitory synapses with respect to the excitatory ones
    rel_inh_strength=1,
    # Mean firing rate from a single poisson cell (kHz)
    poiss_lambda=1,
    # Seed for poisson spike generators
    seed=42,
)
meters.checkpoint("recipe-create", context)

decomp = arbor.partition_load_balance(recipe, context)

meters.checkpoint("load-balance", context)

sim = arbor.simulation(recipe, context, decomp)
sim.record(arbor.spike_recording.all)

meters.checkpoint("simulation-init", context)

sim.run(100, 1)

meters.checkpoint("simulation-run", context)

# Print profiling information
print(f"{arbor.meter_report(meters, context)}")

# Print spike times
print(f"{len(sim.spikes())} spikes generated.")


df = pd.DataFrame(
    [(cell_id, t) for (cell_id, _), t in sim.spikes()],
    columns=['Cell', 'Time (ms)'])

fig = px.scatter(df, x='Time (ms)', y='Cell', title='Spikes')
fig_html = fig.to_html(include_plotlyjs=False, full_html=False)
arbor_playground.render_html(fig_html)
