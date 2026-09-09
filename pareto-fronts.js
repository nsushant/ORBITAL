(() => {
  const mount = document.querySelector("#pareto-charts");
  if (!mount) return;

  const namespace = "http://www.w3.org/2000/svg";
  const scenarios = [
    { key: "S1_repair", title: "S1 reactive repair" },
    { key: "S2_refuel", title: "S2 planned refuelling" },
    { key: "S3_deorbit", title: "S3 end-of-life deorbit" },
  ];
  const algorithms = [
    { key: "mdls", colour: "#bceaff", marker: "circle" },
    { key: "nsga2", colour: "#62c6ff", marker: "triangle" },
  ];

  function element(name, attributes = {}) {
    const node = document.createElementNS(namespace, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function parseCsv(text) {
    const [header, ...rows] = text.trim().split(/\r?\n/);
    const columns = header.split(",");
    return rows.map((row) => {
      const values = row.split(",").map(Number);
      return Object.fromEntries(columns.map((column, index) => [column, values[index]]));
    });
  }

  async function loadFront(algorithm, scenario) {
    const response = await fetch(`assets/${algorithm}_${scenario}_01.csv`);
    if (!response.ok) throw new Error(`Unable to load ${algorithm} ${scenario}`);
    return parseCsv(await response.text());
  }

  function scale(value, domainMin, domainMax, rangeMin, rangeMax) {
    return rangeMin + ((value - domainMin) / (domainMax - domainMin || 1)) * (rangeMax - rangeMin);
  }

  function label(svg, text, x, y, className, anchor = "middle") {
    const node = element("text", { x, y, class: className, "text-anchor": anchor });
    node.textContent = text;
    svg.appendChild(node);
    return node;
  }

  function marker(svg, point, algorithm, x, y) {
    const common = {
      class: `pareto-point ${algorithm.key}`,
      fill: "none",
      stroke: algorithm.colour,
      "stroke-width": 1.7,
    };
    const node = algorithm.marker === "circle"
      ? element("circle", { ...common, cx: x, cy: y, r: 3.2 })
      : element("path", {
          ...common,
          d: `M ${x} ${y - 4} L ${x + 3.8} ${y + 3.1} L ${x - 3.8} ${y + 3.1} Z`,
        });
    const title = element("title");
    title.textContent = `${algorithm.key === "mdls" ? "MDLS" : "NSGA-II"}: ${Math.round(point.f1_dv).toLocaleString()} m/s, $${(point.f2_unrecovered_value / 1e6).toFixed(1)}M unrecovered`;
    node.appendChild(title);
    svg.appendChild(node);
  }

  function drawPanel(svg, scenario, series, panelIndex, domains) {
    const panelWidth = 340;
    const left = 58 + panelIndex * panelWidth;
    const right = left + 270;
    const top = 54;
    const bottom = 280;
    const { xMax, yMin, yMax } = domains;

    label(svg, scenario.title, (left + right) / 2, 25, "chart-title");

    for (let xValue = 0; xValue <= xMax; xValue += 2500) {
      const x = scale(xValue, 0, xMax, left, right);
      svg.appendChild(element("line", { x1: x, y1: top, x2: x, y2: bottom, class: "chart-grid" }));
      label(svg, Math.round(xValue).toLocaleString(), x, bottom + 20, "chart-tick");
    }
    for (let yValue = yMin; yValue <= yMax; yValue += 20) {
      const y = scale(yValue, yMin, yMax, bottom, top);
      svg.appendChild(element("line", { x1: left, y1: y, x2: right, y2: y, class: "chart-grid" }));
      label(svg, Math.round(yValue), left - 10, y + 3, "chart-tick", "end");
    }

    svg.appendChild(element("line", { x1: left, y1: bottom, x2: right, y2: bottom, class: "chart-axis" }));
    svg.appendChild(element("line", { x1: left, y1: top, x2: left, y2: bottom, class: "chart-axis" }));

    series.forEach(({ algorithm, points }) => {
      points.forEach((point) => {
        const x = scale(point.f1_dv, 0, xMax, left, right);
        const y = scale(point.f2_unrecovered_value / 1e6, yMin, yMax, bottom, top);
        marker(svg, point, algorithm, x, y);
      });
    });

    label(svg, "cumulative ΔV [m/s]", (left + right) / 2, 323, "chart-axis-label");
    if (panelIndex === 0) {
      const yLabel = label(svg, "unrecovered value [$M]", 15, (top + bottom) / 2, "chart-axis-label");
      yLabel.setAttribute("transform", `rotate(-90 15 ${(top + bottom) / 2})`);
    }
  }

  async function render() {
    try {
      const loaded = await Promise.all(
        scenarios.flatMap((scenario) => algorithms.map(async (algorithm) => ({
          scenario: scenario.key,
          algorithm,
          points: await loadFront(algorithm.key, scenario.key),
        }))),
      );
      const svg = element("svg", {
        viewBox: "0 0 1050 340",
        role: "presentation",
        focusable: "false",
      });
      const allPoints = loaded.flatMap((entry) => entry.points);
      const domains = {
        xMax: Math.ceil(Math.max(...allPoints.map((point) => point.f1_dv)) / 2500) * 2500,
        yMin: Math.floor(Math.min(...allPoints.map((point) => point.f2_unrecovered_value / 1e6)) / 20) * 20,
        yMax: Math.ceil(Math.max(...allPoints.map((point) => point.f2_unrecovered_value / 1e6)) / 20) * 20,
      };
      scenarios.forEach((scenario, panelIndex) => {
        drawPanel(
          svg,
          scenario,
          loaded.filter((entry) => entry.scenario === scenario.key),
          panelIndex,
          domains,
        );
      });
      mount.replaceChildren(svg);
    } catch (error) {
      mount.textContent = "The Pareto-front data could not be loaded.";
      mount.classList.add("chart-error");
    }
  }

  render();
})();
