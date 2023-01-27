const MODELS = [
{ title: 'Example models:', is_header: true, },
    {
        title: 'Brunel network',
        url: 'models/brunel.py',
        description: 'Advanced network example. Sparsely connected excitatory and inhibitory LIF cells exhibit different synchronization states. Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. Journal of computational neuroscience, 8(3), 183-208.',
        enabled: true
    },
    {
        title: 'Cluster synchronization in the Inferior Olive',
        url: 'models/io_network.py',
        description: 'Inferior Olive neuron model due to Smol et al. Uses a custom catalogue generated from NeuroML source using NMLCC. Network version.',
        enabled: true,
        filesystem: [
            { path: 'io-catalogue.so', url: 'catalogue/io-catalogue.so' }
        ],
    },
{ title: 'Arbor usage examples:', is_header: true, },
    {
        load_first: true,
        title: 'Single Hodgkin-Huxley cell (single_cell_model)',
        url: 'models/single_cell_model.py',
        description: 'The smallest possible useful Arbor example. A single Hodgkin-Huxley cell constructed via the simple arbor.single_cell_model API. The single_cell_model is not as powerful a full recipe construction, but is very convenient for if you\'re only simulating a single cell and only interested in recording voltages.',
        enabled: true
    },
    {
        title: 'Single Hodgkin-Huxley cell (recipe)',
        url: 'models/single_cell_recipe.py',
        description: 'Single Hodgkin-Huxley cell via the recipe API, which allows for more complicated network construction.',
        enabled: true
    },
    {
        title: 'Single cell Allen',
        url: 'models/single_cell_allen.py',
        description: 'Multicompartmental cell comparison between NEURON and Arbor. Quite slow to run.',
        filesystem: [
            { path: 'single_cell_allen_fit.json', url: 'models/single_cell_allen_fit.json' },
            { path: 'single_cell_allen.swc', url: 'models/single_cell_allen.swc' },
            { path: 'single_cell_allen_neuron_ref.csv', url: 'https://raw.githubusercontent.com/arbor-sim/arbor/master/python/example/single_cell_allen_neuron_ref.csv' }
        ],
        enabled: false
    },
    {
        title: 'Single cell detailed recipe (SWC morphology)',
        url: 'models/single_cell_detailed_recipe.py',
        description: 'Advanced single cell example that loads the cell morphology from an external SWC file. Sodium concentration reversal potential is calculated using the Nernst equations. This example is known to break on firefox.',
        filesystem: [
            { path: 'single_cell_detailed.swc', url: 'models/single_cell_detailed.swc' }
        ],
        enabled: true
    },
    {
        title: 'Single Inferior Olive Neuron (custom mechanisms)',
        url: 'models/io_single_cell.py',
        description: 'Inferior Olive neuron model due to Smol et al. Uses a custom catalogue generated from NeuroML source using NMLCC.',
        enabled: true,
        filesystem: [
            { path: 'io-catalogue.so', url: 'catalogue/io-catalogue.so' }
        ],
    },
    {
        title: 'Ring network',
        url: 'models/network_ring.py',
        description: 'Minimal example of a network in Arbor. Four cells connected with delayed synapses leads to a persistent traveling wave in the network.',
        enabled: true
    },
    {
        title: 'Spike-timing-dependent plasticity',
        url: 'models/single_cell_stdp.py',
        description: 'STDP example using a single cell and explicit spike generators. Plots out weight change as a function of simulus distance over multiple simulations.',
        enabled: true
    },
    {
        title: 'Gap junction network',
        url: 'models/gap_junctions.py',
        description: 'Minimal example of multicompartmental cells connected via electrically conducting gap junctions and time delay synapses.',
        enabled: true
    },
    {
        title: 'Ion diffusion',
        url: 'models/diffusion.py',
        description: 'Minimal example showcasing sodium diffusion through cell compartments.',
        enabled: true
    },
    {
        title: 'Modifying network topology',
        url: 'models/plasticity.py',
        description: 'Example of modifying synaptic connections in a running simulation. This is done via the simulation.update_connections(recipe) function, which re-reads the recipe\'s connections_on().',
        enabled: true
    },
]

function quote(text) {
    return (''+text).replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function add_resize_handler() {
    const grid_parent = document.getElementsByClassName('parent')[0]
    const DRAG_NONE = 0,
          DRAG_HOR = 1,
          DRAG_VER = 2;
    let drag_state = DRAG_NONE;
    let on_start = (e) => {
        if (e.target.className === 'hresize') {
            drag_state = DRAG_HOR
            e.preventDefault()
        }
        if (e.target.className === 'vresize') {
            drag_state = DRAG_VER
            e.preventDefault()
        }
    }
    function get_total_px(gridspec) {
        let total = 0
        for (let m of gridspec.matchAll(/([0-9]+)px/g)) {
            total += parseInt(m[1])
        }
        return total
    }
    let on_move = (e) => {
        if (drag_state == DRAG_HOR) {
            let x = e.changedTouches === undefined ? e.clientX : e.changedTouches[0].clientX
            let rect = grid_parent.getBoundingClientRect()
            let fraction = (x - rect.left) / (rect.width - get_total_px(grid_parent.style.gridTemplateColumns));
            fraction = Math.min(Math.max(.05, fraction), .95)
            const total = 1000
            fraction = Math.round(fraction * total)
            grid_parent.style.gridTemplateColumns = `${fraction}fr 10px ${total - fraction}fr` // <-- keep in sync with css
            e.preventDefault()
        }
        if (drag_state == DRAG_VER) {
            let y = e.changedTouches === undefined ? e.clientY : e.changedTouches[0].clientY
            let rect = grid_parent.getBoundingClientRect()
            let fraction = (y - rect.top - 60) / (rect.height - get_total_px(grid_parent.style.gridTemplateRows));
            fraction = Math.min(Math.max(.05, fraction), .95)
            const total = 1000
            fraction = Math.round(fraction * total)
            console.log(fraction)
            grid_parent.style.gridTemplateRows = `60px ${fraction}fr 10px ${total - fraction}fr 10px` // <-- keep in sync with css
            e.preventDefault()
        }
    }
    let on_end = (e) => {
        drag_state = DRAG_NONE
    }
    document.addEventListener('mousedown', on_start);
    document.addEventListener('mousemove', on_move);
    document.addEventListener('mouseup', on_end)
    document.addEventListener('touchstart', on_start, { passive: false });
    document.addEventListener('touchmove', on_move, { passive: false });
    document.addEventListener('touchend', on_end, { passive: false })
}

async function main() {
    add_resize_handler();
    let editor = null
    let loader_icon = document.getElementById('loader-icon')
    let console_output = document.getElementById('console')
    let run_btn = document.getElementById('run-btn')
    let welcome_btn = document.getElementById('welcome-btn')
    let current_modal = null
    loader_icon.classList.add('loading')
    function message_ok(msg) {
        console_output.innerHTML += quote(msg) + '\n'
    }
    function message_err(msg) {
        console_output.innerHTML += '<span class="error">' + quote(msg) + '</span>\n'
    }
    /* START MODAL CODE */
    document.querySelectorAll('.modal-close').forEach(e => {
        e.onclick = function() {
            if (current_modal != null) {
                current_modal.style.display = 'none';
                current_modal = null
            }
        }
    })
    function show_modal(id) {
        let m = document.getElementById(id)
        current_modal = m
        m.style.display = 'block'
    }
    window.onclick = function(event) {
        if (event.target == current_modal) {
            if (current_modal != null) {
                current_modal.style.display = 'none';
            }
        }
    }
    welcome_btn.onclick = () => {
        show_modal('welcome-modal')
    }
    let container = document.getElementById('available-models')
    MODELS.forEach((model, i) => {
        // const is_localhost = location.hostname === 'localhost' || location.hostname === '127.0.0.1'
        // if (!is_localhost && !model.enabled) { return }
        if (model.is_header) {
            container.innerHTML += `
                <h3 class="loadable-models-header">${quote(model.title)}</h3>
            `
        } else {
            let possible_file_list = ''
            if (model.filesystem) {
                let lis = model.filesystem.map(({path}) => `<code>${quote(path)}</code>`).join(', ')
                possible_file_list = `<p>Extra: ${lis}</p>`
            }
            container.innerHTML += `
                <div class="loadable-model" data-model-idx="${i}">
                    <h3>${quote(model.title)}</h3>
                    <p>${quote(model.description)}</p>
                    ${possible_file_list}
                </div>
            `
        }
    })
    async function load_model(model) {
        if (editor == null) return
        loader_icon.classList.add('loading')
        console_output.innerText = ''
        render_html_output('')
        let res = await fetch(model.url)
        editor.session.setValue(await res.text())
        if (model.filesystem) {
            pyodide.FS.chdir('/home/pyodide')
            for (const {path, url} of model.filesystem) {
                let r = await fetch(url)
                let data = await r.blob() // text() fails for .so files
                console.log()
                pyodide.FS.writeFile(path, new Uint8Array(await data.arrayBuffer()));
                message_ok('Created file "' + path + '"')
            }
        }

        if (model.enabled) {
            await run_code()
        } else {
            message_ok('Note: script not automatically executed')
        }
        loader_icon.classList.remove('loading')
    }
    document.querySelectorAll('.loadable-model').forEach(target => {
        target.onclick = async () => {
            if (editor == null) return
            let idx = target.getAttribute('data-model-idx')
            let model = MODELS[idx]
            await load_model(model)
            if (current_modal != null) {
                current_modal.style.display = 'none';
            }
        }
    })

    /* END MODAL CODE */
    message_ok('Loading...')
    console.time('loadPyodide')
    let pyodide = await loadPyodide({
        stdout: (msg) => {
            message_ok(msg)
        },
        stderr: (msg) => {
            message_err(msg)
        },
    })
    console.timeEnd('loadPyodide')
    py = pyodide // global export for debugging
    console.time('loadPackages')
    await pyodide.loadPackage([
        'micropip',
        'numpy',
        'pandas',
        (await (await fetch('ARBOR_WHEEL_NAME')).text()).trim()
    ])
    const micropip = pyodide.pyimport('micropip');
    await micropip.install('plotly');
    console.timeEnd('loadPackages')

    function render_html_output(html) {
        var range = document.createRange();
        let container = document.getElementById('render-html-output')
        range.selectNode(container);
        var documentFragment = range.createContextualFragment(html);
        while (container.hasChildNodes()) {  
            container.removeChild(container.firstChild);
        }
        container.appendChild(documentFragment);
        container.className = "plotly";
    }
    let plot_module = {
        render_html(html) {
            render_html_output(html)
        }
    }
    console.time('registerJsModule')
    pyodide.registerJsModule('arbor_playground', plot_module)
    message_ok('Registered html render module')
    console.timeEnd('registerJsModule')

    async function run_code(code=null) {
        if (code == null) {
            console.time('run_code')
            console_output.innerText = ''
            render_html_output('')
            run_btn.classList.remove("ready");
            message_ok('Console output [' + (new Date()).toISOString() + ']')
        }
        try {
            pyodide.globals.set('code_to_run',
                code == null ? editor.getValue() : code);
            await pyodide.runPython('exec(compile(code_to_run, "main.py", "exec"), {})')
        } catch (error) {
            let test = '' + error
            console.log(test)
            if (test.indexOf('PythonError') === -1) {
                // probably internal pyodide error or CppError
                // this means we can't use any of the pyodide functions anymore..
                message_err(test)
            } else {
                const traceback = pyodide.pyimport('traceback')
                const lines = traceback.format_exception(error)
                let frames = []
                for (let i = 0; i < lines.__len__(); i++) {
                    frames.push(...(lines.get(i).split('\n')))
                }
                let filtered_frames = ['PythonError: Traceback (most recent call last)']
                let skip = true
                for (let i = 1; i < frames.length; i++) {
                    if (frames[i].indexOf("main.py") !== -1) {
                        skip = false
                    }
                    if (!skip) {
                        filtered_frames.push(frames[i])
                    }
                }
                message_err(filtered_frames.join('\n'))
            }
        }
        console_output.scrollTop = console_output.scrollHeight;
        if (code == null) {
            run_btn.classList.add("ready");
            console.timeEnd('run_code')
        }
    }

    console.time('cache_imports')
    await run_code('import pandas, arbor, plotly, numpy')
    console.timeEnd('cache_imports')

    message_ok('Cached pandas, arbor, plotly')

    console.time('aceEditor')
    editor = ace.edit('editor')
    editor.setTheme('ace/theme/monokai')
    editor.session.setMode('ace/mode/python')
    message_ok('Set up editor')
    console.timeEnd('aceEditor')

    message_ok('Ready!')

    run_btn.onclick = async () => {
        await run_code();
    }

    await load_model(MODELS.find(m => m.load_first))

    loader_icon.classList.remove('loading')
    run_btn.classList.add("ready");

    show_modal('welcome-modal')
};

main();
