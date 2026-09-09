(() => {
  const scenarios = window.DEPOT_DATA;
  const labels = { repair: "S1 reactive repair", refuel: "S2 planned refuelling", deorbit: "S3 end-of-life deorbit" };
  const budget = 2000;
  const lossCap = 0.4;
  const svg = document.querySelector("#depot-map");
  const table = document.querySelector("#depot-table");
  const buttons = [...document.querySelectorAll(".scenario-button")];
  const resetButton = document.querySelector("#reset-architecture");
  const status = document.querySelector("#feasibility-status");
  const statusTitle = status.querySelector("strong");
  const statusDetail = document.querySelector("#feasibility-detail");
  const ns = "http://www.w3.org/2000/svg";
  const plot = { left: 72, right: 688, top: 24, bottom: 360 };
  const state = {};
  let activeScenario = "repair";
  let dragIndex = null;

  Object.entries(scenarios).forEach(([key, data]) => {
    state[key] = data.original.map((depot) => ({ ...depot, moved: false }));
  });

  function svgNode(name, attributes = {}, text = "") {
    const node = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    return node;
  }

  function xScale(altitude) {
    return plot.left + ((altitude - 400) / 800) * (plot.right - plot.left);
  }

  function yScale(inclination) {
    return plot.bottom - ((inclination - 53) / 44) * (plot.bottom - plot.top);
  }

  function fromPointer(event) {
    const box = svg.getBoundingClientRect();
    return { x: (event.clientX - box.left) * 720 / box.width, y: (event.clientY - box.top) * 430 / box.height };
  }

  function decodeCoverage(depot, n) {
    if (depot.decoded) return depot.decoded;
    const bytes = Uint8Array.from(atob(depot.cov), (character) => character.charCodeAt(0));
    depot.decoded = Array.from({ length: n }, (_, index) => Boolean(bytes[index >> 3] & (1 << (index & 7))));
    return depot.decoded;
  }

  function nearestLiveLocation(point, data) {
    let best = null;
    let bestDistance = Infinity;
    data.locations.forEach((location) => {
      if (!location.live) return;
      const distance = (xScale(location.alt) - point.x) ** 2 + (yScale(location.i) - point.y) ** 2;
      if (distance < bestDistance) {
        best = location;
        bestDistance = distance;
      }
    });
    return best;
  }

  function architectureMetrics(data, depots) {
    const covered = Array(data.n).fill(false);
    depots.forEach((depot) => decodeCoverage(depot, data.n).forEach((value, index) => { covered[index] ||= value; }));
    const totals = { starlink: 0, planet: 0 };
    const recovered = { starlink: 0, planet: 0 };
    let requests = 0;
    data.values.forEach((value, index) => {
      const group = data.groups[index];
      totals[group] += value;
      if (covered[index]) {
        recovered[group] += value;
        requests += data.nreq[index];
      }
    });
    const cost = depots.reduce((sum, depot) => sum + depot.cost, 0);
    const fleet = depots.reduce((sum, depot) => sum + depot.fleet, 0);
    const dv = depots.reduce((sum, depot) => sum + depot.dv, 0);
    const slLoss = 1 - recovered.starlink / totals.starlink;
    const plLoss = 1 - recovered.planet / totals.planet;
    return {
      cost, fleet, dv, requests, slLoss, plLoss,
      recovered: (recovered.starlink + recovered.planet) / 1e6,
      feasible: cost <= budget + 1e-6 && slLoss <= lossCap + 1e-9 && plLoss <= lossCap + 1e-9,
    };
  }

  function drawBaseMap(data) {
    svg.replaceChildren(
      svgNode("title", { id: "map-title" }, `${labels[activeScenario]} depot locations`),
      svgNode("desc", { id: "map-description" }, "Drag numbered depot markers among evaluated altitude and inclination grid locations."),
    );
    [400, 600, 800, 1000, 1200].forEach((altitude) => {
      const x = xScale(altitude);
      svg.append(
        svgNode("line", { x1: x, y1: plot.top, x2: x, y2: plot.bottom, stroke: "rgba(169,204,232,.2)", "stroke-width": 1 }),
        svgNode("text", { x, y: 388, fill: "#a9cce8", "font-size": 13, "text-anchor": "middle" }, altitude),
      );
    });
    [53, 60, 70, 80, 90, 97].forEach((inclination) => {
      const y = yScale(inclination);
      svg.append(
        svgNode("line", { x1: plot.left, y1: y, x2: plot.right, y2: y, stroke: "rgba(169,204,232,.2)", "stroke-width": 1 }),
        svgNode("text", { x: 58, y: y + 4, fill: "#a9cce8", "font-size": 13, "text-anchor": "end" }, `${inclination}°`),
      );
    });
    data.locations.forEach((location) => {
      svg.append(svgNode("circle", {
        cx: xScale(location.alt), cy: yScale(location.i), r: location.live ? 2.15 : 1.25,
        fill: location.live ? "rgba(169,204,232,.38)" : "rgba(169,204,232,.1)",
      }));
    });
    svg.append(
      svgNode("text", { x: 380, y: 422, fill: "#eef8ff", "font-size": 14, "text-anchor": "middle" }, "Depot altitude [km]"),
      svgNode("text", { x: 17, y: 194, fill: "#eef8ff", "font-size": 14, "text-anchor": "middle", transform: "rotate(-90 17 194)" }, "Inclination [deg]"),
    );
  }

  function drawGhosts(data) {
    const ghosts = svgNode("g", { class: "optimal-ghosts" });
    data.original.forEach((depot) => ghosts.append(svgNode("circle", {
      cx: xScale(depot.alt), cy: yScale(depot.i), r: 21, fill: "none",
      stroke: "rgba(188,234,255,.48)", "stroke-width": 1.5, "stroke-dasharray": "3 5",
    })));
    svg.append(ghosts);
  }

  function moveDepot(index, location) {
    if (!location) return;
    state[activeScenario][index] = { ...location, moved: true };
    render(activeScenario);
  }

  function keyboardMove(event, index) {
    const steps = { ArrowLeft: [-50, 0], ArrowRight: [50, 0], ArrowUp: [0, 2], ArrowDown: [0, -2] };
    if (!steps[event.key]) return;
    event.preventDefault();
    const depot = state[activeScenario][index];
    const [da, di] = steps[event.key];
    const target = nearestLiveLocation({ x: xScale(depot.alt + da), y: yScale(depot.i + di) }, scenarios[activeScenario]);
    moveDepot(index, target);
  }

  function drawSelection(data, depots) {
    const group = svgNode("g", { class: "selected-depots" });
    depots.forEach((depot, index) => {
      const marker = svgNode("g", {
        class: "selected-depot", transform: `translate(${xScale(depot.alt)} ${yScale(depot.i)})`, tabindex: 0,
        role: "button", "aria-label": `Move depot ${index + 1}, currently ${depot.alt} kilometre altitude and ${depot.i} degree inclination`,
      });
      marker.append(
        svgNode("circle", { r: 17, fill: depot.moved ? "#0b65a5" : "rgba(3,29,61,.96)", stroke: "#bceaff", "stroke-width": 2 }),
        svgNode("circle", { r: 27, fill: "none", stroke: "rgba(98,198,255,.25)", "stroke-width": 1 }),
        svgNode("text", { y: 5, fill: "#eef8ff", "font-size": 14, "font-weight": 700, "text-anchor": "middle", "pointer-events": "none" }, index + 1),
      );
      marker.addEventListener("pointerdown", (event) => { dragIndex = index; marker.setPointerCapture(event.pointerId); });
      marker.addEventListener("pointermove", (event) => {
        if (dragIndex !== index) return;
        const location = nearestLiveLocation(fromPointer(event), data);
        if (location) marker.setAttribute("transform", `translate(${xScale(location.alt)} ${yScale(location.i)})`);
      });
      marker.addEventListener("pointerup", (event) => {
        if (dragIndex !== index) return;
        dragIndex = null;
        moveDepot(index, nearestLiveLocation(fromPointer(event), data));
      });
      marker.addEventListener("pointercancel", () => { dragIndex = null; render(activeScenario); });
      marker.addEventListener("keydown", (event) => keyboardMove(event, index));
      group.append(marker);
    });
    svg.append(group);
  }

  function renderTable(depots) {
    table.innerHTML = depots.map((depot, index) => `
      <tr>
        <td><span class="depot-index">${index + 1}</span>${depot.moved ? '<span class="row-state">moved</span>' : ""}</td>
        <td>${depot.alt} km / ${depot.i}°</td>
        <td>${depot.fleet}</td>
        <td>${depot.dv.toLocaleString()} m/s</td>
        <td>$${depot.cost.toFixed(1)}M</td>
        <td>$${depot.value.toFixed(1)}M</td>
      </tr>
    `).join("");
  }

  function renderStatus(metrics, moved) {
    status.classList.toggle("is-feasible", metrics.feasible);
    status.classList.toggle("is-infeasible", !metrics.feasible);
    statusTitle.textContent = metrics.feasible ? "Feasible architecture" : "Constraints violated";
    if (!moved) {
      statusDetail.textContent = "Published lexicographic optimum";
      return;
    }
    const issues = [];
    if (metrics.cost > budget) issues.push(`budget exceeded by $${(metrics.cost - budget).toFixed(1)}M`);
    if (metrics.slLoss > lossCap) issues.push(`Starlink loss ${(100 * metrics.slLoss).toFixed(1)}%`);
    if (metrics.plLoss > lossCap) issues.push(`Planet Labs loss ${(100 * metrics.plLoss).toFixed(1)}%`);
    statusDetail.textContent = issues.length ? issues.join(" · ") :
      `Starlink loss ${(100 * metrics.slLoss).toFixed(1)}% · Planet Labs loss ${(100 * metrics.plLoss).toFixed(1)}%`;
  }

  function render(key) {
    activeScenario = key;
    const data = scenarios[key];
    const depots = state[key];
    const metrics = architectureMetrics(data, depots);
    buttons.forEach((button) => {
      const active = button.dataset.scenario === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelector("#summary-depots").textContent = depots.length;
    document.querySelector("#summary-fleet").textContent = metrics.fleet;
    document.querySelector("#summary-cost").textContent = `$${(metrics.cost / 1000).toFixed(2)}B`;
    document.querySelector("#summary-value").textContent = `$${metrics.recovered.toFixed(1)}M`;
    drawBaseMap(data);
    drawGhosts(data);
    drawSelection(data, depots);
    renderTable(depots);
    renderStatus(metrics, depots.some((depot) => depot.moved));
  }

  buttons.forEach((button) => button.addEventListener("click", () => render(button.dataset.scenario)));
  resetButton.addEventListener("click", () => {
    state[activeScenario] = scenarios[activeScenario].original.map((depot) => ({ ...depot, moved: false }));
    render(activeScenario);
  });
  render("repair");
})();
