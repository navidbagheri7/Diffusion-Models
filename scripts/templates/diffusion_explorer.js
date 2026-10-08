"use strict";
const evidence = JSON.parse(document.getElementById("evidence").textContent);
const groups = ["Forward noising", "Prediction targets", "Scores and reverse dynamics", "Flows and transport", "Bridges and interpolants", "Solvers and few-step models", "Image generators", "Representations and DiT", "Guidance and text", "GAN objectives", "Likelihood and evaluation"];
// Forward noising and prediction are separate live views; nine evidence areas share the equation panel.
const $ = id => document.getElementById(id);
let group = groups[0];

function node(tag, text) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    return e;
}

function fmt(v) {
    return v === undefined || v === null ? "—" : typeof v === "number" ? (Math.abs(v) > 1e5 || Math.abs(v) < .001 && v !== 0 ? v.toExponential(3) : v.toFixed(4)) : String(v);
}

function coeff(kind, t) {
    if (kind === "ve") return [1, .01 * Math.pow(2000, t)];
    if (kind === "vp_logsnr") {
        let l = 15 - 30 * t;
        return [Math.sqrt(1 / (1 + Math.exp(-l))), Math.sqrt(1 / (1 + Math.exp(l)))];
    }
    if (kind === "vp_cosine") {
        let a = Math.cos((t + .008) / 1.008 * Math.PI / 2) / Math.cos(.008 / 1.008 * Math.PI / 2);
        return [a, Math.sqrt(Math.max(0, 1 - a * a))];
    }
    let beta = kind === "vp_exponential" ? 20 * t : .1 * t + .5 * 19.9 * t * t,
        a = Math.exp(-beta / 2),
        variance = -Math.expm1(-beta);
    return [a, kind === "sub_vp" ? variance : Math.sqrt(variance)];
}

function mixture(x, a, s) {
    const variance = a * a * evidence.std ** 2 + s * s;
    const logs = evidence.means.map(mu => -((x[0] - a * mu[0]) ** 2 + (x[1] - a * mu[1]) ** 2) / (2 * variance));
    let max = Math.max(...logs),
        sum = logs.reduce((v, l) => v + Math.exp(l - max), 0),
        weights = logs.map(l => Math.exp(l - max) / sum);
    let score = [0, 0];
    weights.forEach((w, i) => {
        for (let j = 0; j < 2; j++) score[j] -= w * (x[j] - a * evidence.means[i][j]) / variance;
    });
    return {
        score,
        posterior: x.map((v, j) => (v + s * s * score[j]) / a),
        logDensity: max + Math.log(sum) - Math.log(8) - Math.log(2 * Math.PI * variance)
    };
}

function random(seed) {
    return () => {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
const rng = random(71);

function normal() {
    return Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-12))) * Math.cos(2 * Math.PI * rng());
}
const clean = Array.from({
        length: 1024
    }, (_, i) => evidence.means[i % 8].map(v => v + .25 * normal())),
    noise = clean.map(() => [normal(), normal()]);

function svg(tag, attrs, text) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text) e.textContent = text;
    return e;
}

function live() {
    let kind = $("path").value,
        t = Number($("time").value),
        [a, s] = coeff(kind, t),
        probe = mixture([1, -.5], a, s);
    window.liveDiff = {
        kind,
        t,
        alpha: a,
        sigma: s,
        ...probe
    };
    $("time-value").textContent = t.toFixed(3);
    $("live-title").textContent = group === "Prediction targets" ? "Same corruption, different training targets" : "Forward Gaussian corruption";
    $("numbers").replaceChildren(...[
        ["Signal α", a],
        ["Noise σ", s],
        ["Variance of one noised component", a * a * .25 ** 2 + s * s],
        ["log-SNR", 2 * Math.log(a / s)]
    ].map(([k, v]) => {
        let e = node("div");
        e.append(node("div", fmt(v)), node("div", k));
        e.firstChild.className = "number";
        e.lastChild.className = "muted";
        return e;
    }));
    $("equations").textContent = group === "Prediction targets" ? "ε = (xₜ − αx₀)/σ; score = −ε/σ. With v = αε − σx₀, x₀ = (αxₜ − σv)/(α²+σ²). The denominator matters for sub-VP and VE. These pointwise targets differ from their conditional regression means." : "∇ log pₜ(x) averages component scores using posterior mixture responsibilities. E[X₀|Xₜ=x] = (x + σ²∇ log pₜ(x))/α. The ODE uses half of g² times the score; the reverse SDE uses the full coefficient.";
    $("probe").textContent = JSON.stringify(probe, null, 2);
    const c = $("particles"),
        ctx = c.getContext("2d"),
        limit = Math.max(5, 3 * a + 3 * s),
        scale = 500 / (2 * limit),
        px = x => 340 + x * scale,
        py = y => 270 - y * scale;
    ctx.clearRect(0, 0, 680, 540);
    ctx.strokeStyle = "#dce7eb";
    ctx.beginPath();
    ctx.moveTo(30, 270);
    ctx.lineTo(650, 270);
    ctx.moveTo(340, 20);
    ctx.lineTo(340, 520);
    ctx.stroke();
    const palette = ["#087f8c", "#c05b79", "#bc8c26", "#5874b6", "#65944a", "#8e62b7", "#cc7549", "#477784"];
    clean.forEach((x, i) => {
        ctx.fillStyle = palette[i % 8];
        ctx.globalAlpha = .5;
        ctx.beginPath();
        ctx.arc(px(a * x[0] + s * noise[i][0]), py(a * x[1] + s * noise[i][1]), 1.6, 0, 2 * Math.PI);
        ctx.fill();
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#183246";
    for (let x = -4; x <= 4; x += 1)
        for (let y = -4; y <= 4; y += 1) {
            let v = mixture([x, y], a, s).score,
                n = Math.hypot(...v),
                length = Math.min(.5, n * .08);
            if (n < 1e-8) continue;
            let dx = v[0] / n * length,
                dy = v[1] / n * length;
            ctx.beginPath();
            ctx.moveTo(px(x), py(y));
            ctx.lineTo(px(x + dx), py(y + dy));
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(px(x + dx), py(y + dy), 1.5, 0, 2 * Math.PI);
            ctx.fillStyle = "#183246";
            ctx.fill();
        }
    ctx.fillStyle = "#567080";
    ctx.font = "12px system-ui";
    ctx.fillText(`x and y range ±${limit.toFixed(2)}; equal axes`, 30, 532);
    $("particle-caption").textContent = "Fixed paired Gaussian draws; colours identify clean mixture components. Score arrow length is capped for readability. The view rescales with noise, so use its declared axes when comparing spread.";
    let chart = svg("svg", {
            viewBox: "0 0 680 230",
            role: "img",
            "aria-label": "Signal and noise coefficients"
        }),
        times = Array.from({
            length: 101
        }, (_, i) => .001 + .998 * i / 100),
        values = times.map(u => coeff(kind, u)),
        maximum = Math.max(1, ...values.flat());
    [0, 1].forEach((k) => chart.append(svg("path", {
        d: times.map((u, i) => `${i?"L":"M"}${50+580*u},${185-150*values[i][k]/maximum}`).join(" "),
        fill: "none",
        stroke: k ? "#bc8c26" : "#087f8c",
        "stroke-width": 2
    })));
    chart.append(svg("text", {
        x: 50,
        y: 20,
        fill: "#087f8c",
        "font-size": 13
    }, "α: signal"), svg("text", {
        x: 180,
        y: 20,
        fill: "#bc8c26",
        "font-size": 13
    }, "σ: noise"), svg("text", {
        x: 50,
        y: 210,
        fill: "#567080",
        "font-size": 13
    }, `Forward time 0 → 1 · coefficient vertical range 0–${maximum.toFixed(2)}`));
    $("schedule-chart").replaceChildren(chart);
}

function selected() {
    return evidence.rows.filter(r => r.group === group);
}

// Reports include scalar dictionaries, analytic trace results, and per-class
// lists. Preserve their measurements and estimator identities without treating
// likelihood bounds, image distances, and caption scores as interchangeable.
function measurementRecords(row) {
    const value = row.metrics;
    const parent = Array.isArray(value) ? {} : value;
    const children = Array.isArray(value) ? value : Array.isArray(value.results) ? value.results : [value];
    return children.map((child, index) => {
        const metrics = {
            ...parent,
            ...child
        };
        delete metrics.results;
        const identity = children.length > 1 ? [child.trace, child.guidance, child.class === undefined ? null : `class ${child.class}`].filter(x => x !== null && x !== undefined).join(" · ") || `result ${index + 1}` : "";
        return {
            name: row.name + (identity ? " · " + identity : ""),
            sourceName: row.name,
            index,
            metrics
        };
    });
}

const comparisonFields = [
    ["nfe", "Sampling NFE/sample", ["nfe_per_sample", "nfe_per_image"]],
    ["calls", "Likelihood field/model calls", ["field_calls", "network_calls"]],
    ["probes", "Trace probes", ["probe_count", "probes"]],
    ["log_density", "Mean log density (nats)", ["mean_log_density_nats"]],
    ["density_bpd", "Dequantized density estimate (bits/dim)", ["dequantized_density_bpd_estimate"]],
    ["elbo_bpd", "Negative ELBO estimate (bits/dim)", ["negative_elbo_bpd_mean"]],
    ["density_rmse", "Log-density RMSE", ["rmse_log_density"]],
    ["density_mae", "Log-density MAE", ["mean_absolute_error"]],
    ["mmd", "MMD² (declared target)", ["mmd_squared_unbiased"]],
    ["w2", "Sliced W₂ (declared target)", ["sliced_w2"]],
    ["fid", "FID (Inception features)", ["fid_2048", "fid", "frechet_inception_distance"]],
    ["kid", "KID mean", ["kid_mean"]],
    ["digit", "Digit feature distance", ["frechet_digit_features_128"]],
    ["accuracy", "Requested class accuracy", ["requested_class_accuracy"]],
    ["clip", "Paired CLIP cosine", ["clip_cosine_mean"]],
    ["clip_margin", "Paired − mismatched cosine", ["paired_minus_mismatched_mean"]],
    ["coverage", "Mode coverage (3σ)", ["mode_coverage_3sigma"]],
    ["samples", "Evaluated samples", ["generated_count", "sample_count", "samples"]],
    ["reference", "Reference samples", ["real_count", "reference_count"]]
];

function fieldValue(metrics, aliases) {
    for (const key of aliases) {
        if (typeof metrics[key] === "number" && Number.isFinite(metrics[key])) return metrics[key];
    }
    return undefined;
}

function measurementFamily(metrics) {
    if (typeof metrics.rmse_log_density === "number") return "Analytic log-density verification";
    if (typeof metrics.negative_elbo_bpd_mean === "number") return "Variational bounds";
    if (typeof metrics.mean_log_density_nats === "number") return "Numerical continuous-density estimates";
    if (typeof metrics.clip_cosine_mean === "number") return "Caption alignment";
    if (typeof metrics.frechet_digit_features_128 === "number") return "Digit classifier features";
    if (fieldValue(metrics, ["fid_2048", "fid", "frechet_inception_distance"]) !== undefined) return "Inception image features";
    if (typeof metrics.mmd_squared_unbiased === "number") return "Distribution distances";
    if (typeof metrics.requested_class_accuracy === "number") return "Class assessment only";
    return null;
}

function measurementDataset(record) {
    if (record.metrics.dataset) return record.metrics.dataset;
    if (record.sourceName.startsWith("Historical / gmm_") || record.metrics.config?.components === 8) return "ring_gmm";
    const prefix = record.sourceName.split("/")[0];
    return ["mnist", "cifar10", "faces", "imagenet", "text", "toy", "bridges"].includes(prefix) ? prefix : "See source dataset";
}

function details(row) {
    const entries = [];
    const identifiers = ["dataset", "trace", "solver", "decoder", "estimator", "guidance", "class", "conditioning_scope", "dequantization"];
    for (const record of measurementRecords(row)) {
        for (const [key, value] of Object.entries(record.metrics)) {
            if (typeof value === "number" || typeof value === "boolean" || identifiers.includes(key) && typeof value === "string") {
                const tr = node("tr");
                tr.dataset.sourceName = record.sourceName;
                tr.dataset.recordIndex = record.index;
                tr.dataset.field = key;
                tr.append(node("th", (record.name === row.name ? "" : record.name.split(row.name + " · ")[1] + " / ") + key.replaceAll("_", " ")), node("td", fmt(value)));
                entries.push(tr);
            }
        }
    }
    $("metrics").replaceChildren(...entries);
    if (!entries.length) {
        const tr = node("tr");
        tr.append(node("td", "Illustrative lesson: no fitted scalar measurements. Read its caption and source."));
        $("metrics").append(tr);
    }
}

function render() {
    const row = selected().find(r => r.name === $("preset").value),
        file = $("view").value;
    $("title").textContent = row.name;
    $("kind").textContent = row.kind;
    $("media").src = file;
    $("media").alt = row.name + " / " + file.split("/").pop();
    $("caption").textContent = file + " · Original recorded view; refer to its lesson caption and metric scope.";
    let metrics = row.metrics;
    $("scope").textContent = metrics.experiment_validity || metrics.scope || metrics.interpretation || metrics.representation || metrics.status || "Recorded experiment/lesson. Training budget, checkpoint identity, reference split and sample counts remain in the source report. This panel preserves exploratory and weak results alongside final studies.";
    $("raw").textContent = JSON.stringify(metrics, null, 2);
    details(row);
    $("source").href = row.source;
    comparison();
}

function setViews() {
    const row = selected().find(r => r.name === $("preset").value);
    $("view").replaceChildren(...row.media.map(v => {
        let e = node("option", v.split("/").slice(-2).join(" / "));
        e.value = v;
        return e;
    }));
    render();
}

function comparison() {
    const records = selected().flatMap(measurementRecords);
    const measured = records.filter(r => measurementFamily(r.metrics));
    $("comparison").replaceChildren(node("p", "Measurements are grouped by recorded dataset and metric family. Different conditioning, sample counts, representations and estimators remain different protocols; these tables do not establish a combined ranking. — means that quantity was not recorded for this experiment.", "muted"));
    if (!measured.length) {
        $("comparison").append(node("p", "This concept contains illustrative lessons, with no fitted comparison metrics. Its figures and analytic controls are available above.", "notice"));
        return;
    }
    const sections = new Map();
    for (const record of measured) {
        const key = measurementDataset(record) + " · " + measurementFamily(record.metrics);
        if (!sections.has(key)) sections.set(key, []);
        sections.get(key).push(record);
    }
    for (const [title, rows] of sections) {
        $("comparison").append(node("h3", title));
        const columns = comparisonFields.filter(([, , aliases]) => rows.some(r => fieldValue(r.metrics, aliases) !== undefined));
        const table = node("table"),
            head = node("tr");
        head.append(node("th", "Recorded experiment"));
        columns.forEach(([, label]) => head.append(node("th", label)));
        table.append(head);
        rows.forEach(r => {
            const tr = node("tr");
            tr.dataset.sourceName = r.sourceName;
            tr.dataset.recordIndex = r.index;
            tr.append(node("td", r.name));
            columns.forEach(([key, , aliases]) => {
                const td = node("td", fmt(fieldValue(r.metrics, aliases)));
                td.dataset.metric = key;
                tr.append(td);
            });
            table.append(tr);
        });
        $("comparison").append(table);
    }
}

function select(value) {
    group = value;
    document.querySelectorAll("nav button").forEach(e => e.setAttribute("aria-selected", String(e.textContent === group)));
    const analytic = group === "Forward noising" || group === "Prediction targets";
    $("analytic").classList.toggle("hidden", !analytic);
    $("recorded").classList.toggle("hidden", analytic);
    if (analytic) {
        live();
        return;
    }
    $("preset").replaceChildren(...selected().map(r => {
        let e = node("option", r.name);
        e.value = r.name;
        return e;
    }));
    setViews();
}
groups.forEach(g => {
    let b = node("button", g);
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => select(g));
    $("concepts").append(b);
});
$("preset").addEventListener("change", setViews);
$("view").addEventListener("change", render);
$("path").addEventListener("change", live);
$("time").addEventListener("input", live);
select(groups[0]);
