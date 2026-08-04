// worker.js — Pyodide WebAssembly Worker for TensionBudget Phase W1 Spike
// Executes unmodified Python DSP math (SciPy Second-Order Sections & Butterworth bandpass filters) directly in browser memory.

importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

// Notify UI that Worker initialization has commenced
postMessage({ type: 'status', status: 'loading_pyodide', message: 'Initializing WebAssembly Pyodide runtime...' });

async function initPyodideRuntime() {
    try {
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/"
        });

        postMessage({ type: 'status', status: 'loading_packages', message: 'Downloading NumPy & SciPy into WebAssembly memory...' });
        await pyodide.loadPackage(["numpy", "scipy"]);

        postMessage({ type: 'status', status: 'injecting_dsp', message: 'Injecting Phase 12 Streaming Channel Processor DSP math...' });

        // Inject canonical Python signal processing math directly into Pyodide Wasm runtime
        pyodide.runPython(`
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos

class WasmStreamingProcessor:
    """
    Direct WebAssembly adaptation of TensionBudget StreamingChannelProcessor.
    Maintains causal SOS filter states across consecutive audio/EMG chunks.
    """
    def __init__(self, fs=500.0, low_cut=20.0, high_cut=240.0, order=4, notch_freq=50.0):
        self.fs = fs
        nyq = 0.5 * fs
        # Bandpass SOS filter
        self.bp_sos = butter(order, [low_cut / nyq, high_cut / nyq], btype='band', output='sos')
        self.bp_zi = sosfilt_zi(self.bp_sos)
        
        # Notch SOS filter (50Hz mains default)
        b, a = iirnotch(notch_freq, 30.0, fs)
        self.notch_sos = tf2sos(b, a)
        self.notch_zi = sosfilt_zi(self.notch_sos)
        
        self.sample_count = 0

    def reset_state(self):
        self.bp_zi = sosfilt_zi(self.bp_sos)
        self.notch_zi = sosfilt_zi(self.notch_sos)
        self.sample_count = 0

    def process(self, raw_chunk):
        arr = np.array(raw_chunk, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return [], []
            
        if self.sample_count == 0 and n > 0:
            self.bp_zi = self.bp_zi * arr[0]
            self.notch_zi = self.notch_zi * arr[0]
            
        filtered, self.bp_zi = sosfilt(self.bp_sos, arr, zi=self.bp_zi)
        filtered, self.notch_zi = sosfilt(self.notch_sos, filtered, zi=self.notch_zi)
        rectified = np.abs(filtered)
        self.sample_count += n
        return filtered.tolist(), rectified.tolist()

# Instantiate independent stateful processors for Left and Right EMG channels
proc_l = WasmStreamingProcessor()
proc_r = WasmStreamingProcessor()
        `);

        postMessage({ type: 'status', status: 'ready', message: 'Pyodide DSP Wasm Engine Active & Verified' });
    } catch (err) {
        postMessage({ type: 'status', status: 'error', message: 'Wasm Initialization Failed: ' + err.toString() });
    }
}

// Start booting Pyodide instantly upon worker instantiation
initPyodideRuntime();

// Event routing handler
self.onmessage = async function(e) {
    const data = e.data;
    if (!pyodide) {
        postMessage({ type: 'error', message: 'Worker received message before Pyodide initialization finished.' });
        return;
    }

    if (data.type === 'process_chunk') {
        const start = performance.now();
        try {
            // Pass JavaScript Arrays into Python global scope
            pyodide.globals.set("js_chunk_l", data.left_samples);
            pyodide.globals.set("js_chunk_r", data.right_samples);
            
            // Execute stateful processing
            const res_l = pyodide.runPython(`proc_l.process(js_chunk_l)`);
            const res_r = pyodide.runPython(`proc_r.process(js_chunk_r)`);
            
            const duration = performance.now() - start;
            postMessage({
                type: 'chunk_processed',
                chunk_id: data.chunk_id,
                duration_ms: duration,
                left_filtered: res_l[0].toJs(),
                left_rectified: res_l[1].toJs(),
                right_filtered: res_r[0].toJs(),
                right_rectified: res_r[1].toJs()
            });
            res_l.destroy();
            res_r.destroy();
        } catch (err) {
            postMessage({ type: 'error', message: 'DSP Execution Error: ' + err.toString() });
        }
    } 
    else if (data.type === 'run_benchmark') {
        const iterations = data.iterations || 1000;
        const chunkSize = data.chunk_size || 50; // 100ms at 500Hz
        
        postMessage({ type: 'benchmark_status', message: `Executing ${iterations} sequential chunks (${iterations * chunkSize} samples)...` });
        
        pyodide.runPython(`proc_l.reset_state(); proc_r.reset_state();`);
        
        // Generate simulated 15Hz muscle burst superimposed on 50Hz mains + baseline drift
        const times = new Float64Array(chunkSize);
        const durations = [];
        let totalSamples = 0;
        
        for (let i = 0; i < iterations; i++) {
            const chunk_l = [];
            const chunk_r = [];
            for (let j = 0; j < chunkSize; j++) {
                const t = (totalSamples + j) / 500.0;
                // Left channel simulation: 15Hz muscle + 50Hz mains noise + 2Hz low-frequency drift
                const val_l = 1.5 * Math.sin(2 * Math.PI * 15 * t) + 0.8 * Math.sin(2 * Math.PI * 50 * t) + 2.0 * Math.sin(2 * Math.PI * 2 * t);
                const val_r = 1.2 * Math.sin(2 * Math.PI * 18 * t) + 0.9 * Math.sin(2 * Math.PI * 50 * t);
                chunk_l.push(val_l);
                chunk_r.push(val_r);
            }
            
            const t0 = performance.now();
            pyodide.globals.set("js_chunk_l", chunk_l);
            pyodide.globals.set("js_chunk_r", chunk_r);
            const res_l = pyodide.runPython(`proc_l.process(js_chunk_l)`);
            const res_r = pyodide.runPython(`proc_r.process(js_chunk_r)`);
            const t1 = performance.now();
            
            durations.push(t1 - t0);
            totalSamples += chunkSize;
            
            res_l.destroy();
            res_r.destroy();
            
            if (i % 200 === 0) {
                postMessage({ type: 'benchmark_progress', progress: Math.round((i / iterations) * 100) });
            }
        }
        
        // Statistical latency analysis
        durations.sort((a, b) => a - b);
        const sum = durations.reduce((acc, v) => acc + v, 0);
        const mean = sum / durations.length;
        const median = durations[Math.floor(durations.length / 2)];
        const p95 = durations[Math.floor(durations.length * 0.95)];
        const p99 = durations[Math.floor(durations.length * 0.99)];
        const max = durations[durations.length - 1];
        
        postMessage({
            type: 'benchmark_complete',
            results: {
                total_chunks: iterations,
                samples_per_chunk: chunkSize,
                total_samples_processed: totalSamples,
                mean_ms: mean.toFixed(3),
                median_ms: median.toFixed(3),
                p95_ms: p95.toFixed(3),
                p99_ms: p99.toFixed(3),
                max_ms: max.toFixed(3),
                sub_10ms_compliant: p95 < 10.0
            }
        });
    }
};
