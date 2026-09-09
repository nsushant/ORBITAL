(() => {
  const earthRadius = 6378.137;
  const scenarios = {
    repair: {
      label: "S1 reactive repair",
      totalFleet: 66,
      totalCost: 1311.2,
      recoveredValue: 204.4,
      depots: [
        { a: 6778.137, i: 53, fleet: 17, dv: 7560, value: 59.7, cost: 335.8 },
        { a: 6928.137, i: 69, fleet: 9, dv: 4127, value: 53.2, cost: 206.6 },
        { a: 6978.137, i: 97, fleet: 23, dv: 10828, value: 87.8, cost: 432.9 },
        { a: 7478.137, i: 97, fleet: 17, dv: 8028, value: 12.0, cost: 335.9 },
      ],
    },
    refuel: {
      label: "S2 planned refuelling",
      totalFleet: 100,
      totalCost: 1921.9,
      recoveredValue: 189.8,
      depots: [
        { a: 6778.137, i: 53, fleet: 19, dv: 8877, value: 72.4, cost: 368.2 },
        { a: 6928.137, i: 69, fleet: 8, dv: 3743, value: 49.5, cost: 190.4 },
        { a: 7078.137, i: 97, fleet: 24, dv: 10688, value: 51.0, cost: 449.0 },
        { a: 7228.137, i: 97, fleet: 24, dv: 11614, value: 32.1, cost: 449.1 },
        { a: 7478.137, i: 97, fleet: 25, dv: 11799, value: 27.5, cost: 465.2 },
      ],
    },
    deorbit: {
      label: "S3 end-of-life deorbit",
      totalFleet: 53,
      totalCost: 1040.0,
      recoveredValue: 216.8,
      depots: [
        { a: 6828.137, i: 53, fleet: 18, dv: 8345, value: 86.5, cost: 352.0 },
        { a: 6878.137, i: 69, fleet: 10, dv: 4610, value: 66.7, cost: 222.7 },
        { a: 7078.137, i: 97, fleet: 25, dv: 11556, value: 63.6, cost: 465.2 },
      ],
    },
  };

  const svg = document.querySelector("#depot-map");
  const table = document.querySelector("#depot-table");
  const buttons = [...document.querySelectorAll(".scenario-button")];
  const ns = "http://www.w3.org/2000/svg";
  const plot = { left: 72, right: 688, top: 24, bottom: 360 };

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

  function drawBaseMap() {
    svg.replaceChildren(
      svgNode("title", { id: "map-title" }, "Selected orbital depot locations"),
      svgNode("desc", { id: "map-description" }, "Candidate depot altitudes and inclinations, with selected locations highlighted for the active scenario."),
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

    for (let altitude = 400; altitude <= 1200; altitude += 50) {
      for (let inclination = 53; inclination <= 97; inclination += 2) {
        svg.append(svgNode("circle", {
          cx: xScale(altitude), cy: yScale(inclination), r: 1.7,
          fill: "rgba(169,204,232,.3)",
        }));
      }
    }

    svg.append(
      svgNode("text", { x: 380, y: 422, fill: "#eef8ff", "font-size": 14, "text-anchor": "middle" }, "Depot altitude [km]"),
      svgNode("text", { x: 17, y: 194, fill: "#eef8ff", "font-size": 14, "text-anchor": "middle", transform: "rotate(-90 17 194)" }, "Inclination [deg]"),
    );
  }

  function drawSelection(data) {
    const group = svgNode("g", { class: "selected-depots" });
    data.depots.forEach((depot, index) => {
      const x = xScale(depot.a - earthRadius);
      const y = yScale(depot.i);
      const marker = svgNode("g", { transform: `translate(${x} ${y})`, tabindex: 0, role: "img", "aria-label": `Depot ${index + 1}: ${Math.round(depot.a - earthRadius)} kilometre altitude, ${depot.i} degree inclination` });
      marker.append(
        svgNode("circle", { r: 17, fill: "rgba(3,29,61,.92)", stroke: "#bceaff", "stroke-width": 2 }),
        svgNode("circle", { r: 25, fill: "none", stroke: "rgba(98,198,255,.28)", "stroke-width": 1 }),
        svgNode("text", { y: 5, fill: "#eef8ff", "font-size": 14, "font-weight": 700, "text-anchor": "middle" }, index + 1),
      );
      group.append(marker);
    });
    svg.append(group);
  }

  function renderTable(data) {
    table.innerHTML = data.depots.map((depot, index) => `
      <tr>
        <td><span class="depot-index">${index + 1}</span></td>
        <td>${Math.round(depot.a - earthRadius)} km / ${depot.i}°</td>
        <td>${depot.fleet}</td>
        <td>${depot.dv.toLocaleString()} m/s</td>
        <td>$${depot.cost.toFixed(1)}M</td>
      </tr>
    `).join("");
  }

  function render(key) {
    const data = scenarios[key];
    buttons.forEach((button) => {
      const active = button.dataset.scenario === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelector("#summary-depots").textContent = data.depots.length;
    document.querySelector("#summary-fleet").textContent = data.totalFleet;
    document.querySelector("#summary-cost").textContent = `$${(data.totalCost / 1000).toFixed(2)}B`;
    document.querySelector("#summary-value").textContent = `$${data.recoveredValue.toFixed(1)}M`;
    drawBaseMap();
    drawSelection(data);
    renderTable(data);
  }

  buttons.forEach((button) => button.addEventListener("click", () => render(button.dataset.scenario)));
  render("repair");
})();
