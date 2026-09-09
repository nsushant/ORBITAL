(() => {
  const canvas = document.querySelector("#orbital-field");
  const context = canvas.getContext("2d");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let animationFrame = 0;

  const shellSpeeds = [0.075, -0.048, 0.034];
  const shellPopulations = [9, 11, 13];
  const constellation = shellPopulations.flatMap((population, orbit) =>
    Array.from({ length: population }, (_, index) => ({
      orbit,
      phase: (index / population) * Math.PI * 2 + orbit * 0.31,
      speed: shellSpeeds[orbit],
      scale: 0.46 + ((index + orbit) % 3) * 0.07,
    })),
  );
  const outerShellSpeeds = [0.021, -0.016];
  const outerConstellation = [10, 12].flatMap((population, orbit) =>
    Array.from({ length: population }, (_, index) => ({
      orbit,
      phase: (index / population) * Math.PI * 2 + orbit * 0.47,
      speed: outerShellSpeeds[orbit],
      scale: 0.42 + ((index + orbit) % 3) * 0.05,
    })),
  );
  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  function geometry() {
    const compact = width < 720;
    return {
      centreX: compact ? width * 0.72 : width * 0.76,
      centreY: compact ? height * 0.34 : height * 0.38,
      tilt: -0.22,
      rings: compact
        ? [[240, 78], [315, 105], [390, 134]]
        : [[390, 128], [520, 172], [650, 220]],
      outerRings: compact
        ? [[520, 174], [650, 218]]
        : [[880, 292], [1100, 368]],
    };
  }

  function pointOnOrbit(ring, angle, layout) {
    const x = Math.cos(angle) * ring[0];
    const y = Math.sin(angle) * ring[1];
    const cosine = Math.cos(layout.tilt);
    const sine = Math.sin(layout.tilt);
    return {
      x: layout.centreX + x * cosine - y * sine,
      y: layout.centreY + x * sine + y * cosine,
    };
  }

  function drawOrbit(ring, layout, opacity = 0.24) {
    context.save();
    context.translate(layout.centreX, layout.centreY);
    context.rotate(layout.tilt);
    context.beginPath();
    context.ellipse(0, 0, ring[0], ring[1], 0, 0, Math.PI * 2);
    context.strokeStyle = `rgba(182, 228, 255, ${opacity})`;
    context.lineWidth = 1;
    context.setLineDash([5, 8]);
    context.stroke();
    context.restore();
  }

  function drawEarth(layout) {
    const radius = width < 720 ? 30 : 44;
    const gradient = context.createRadialGradient(
      layout.centreX - radius * 0.35,
      layout.centreY - radius * 0.35,
      radius * 0.1,
      layout.centreX,
      layout.centreY,
      radius,
    );
    gradient.addColorStop(0, "rgba(45, 151, 218, 0.64)");
    gradient.addColorStop(1, "rgba(3, 37, 79, 0.88)");

    context.save();
    context.translate(layout.centreX, layout.centreY);
    context.beginPath();
    context.arc(0, 0, radius, 0, Math.PI * 2);
    context.fillStyle = gradient;
    context.fill();
    context.strokeStyle = "rgba(175, 229, 255, 0.7)";
    context.lineWidth = 1.2;
    context.stroke();

    context.strokeStyle = "rgba(153, 218, 255, 0.26)";
    context.lineWidth = 0.8;
    [-0.45, 0.45].forEach((offset) => {
      context.beginPath();
      context.ellipse(0, 0, radius * Math.cos(offset), radius, 0, 0, Math.PI * 2);
      context.stroke();
    });
    context.beginPath();
    context.ellipse(0, 0, radius, radius * 0.34, 0, 0, Math.PI * 2);
    context.stroke();

    context.beginPath();
    context.arc(0, 0, radius + 5, 0, Math.PI * 2);
    context.strokeStyle = "rgba(119, 205, 255, 0.16)";
    context.stroke();
    context.restore();
  }

  function drawSatellite(point, scale = 1, active = false, opacity = 1) {
    context.save();
    context.translate(point.x, point.y);
    context.rotate(-0.18);
    context.globalAlpha = opacity;
    context.strokeStyle = active ? "rgba(224, 247, 255, 0.98)" : "rgba(164, 220, 255, 0.76)";
    context.fillStyle = active ? "rgba(98, 198, 255, 0.96)" : "rgba(17, 100, 165, 0.86)";
    context.lineWidth = 1;
    context.fillRect(-5 * scale, -4 * scale, 10 * scale, 8 * scale);
    context.strokeRect(-5 * scale, -4 * scale, 10 * scale, 8 * scale);
    context.fillRect(-18 * scale, -3 * scale, 10 * scale, 6 * scale);
    context.fillRect(8 * scale, -3 * scale, 10 * scale, 6 * scale);
    context.beginPath();
    context.moveTo(0, -4 * scale);
    context.lineTo(4 * scale, -11 * scale);
    context.stroke();
    context.restore();
  }

  function drawDepot(point, opacity = 1, size = 1) {
    const scale = (width < 720 ? 0.72 : 0.9) * size;
    context.save();
    context.translate(point.x, point.y);
    context.rotate(-0.18);
    context.globalAlpha = opacity;
    context.lineWidth = 1;
    context.strokeStyle = "rgba(226, 247, 255, 0.92)";
    context.fillStyle = "rgba(25, 117, 178, 0.92)";

    context.fillRect(-10 * scale, -7 * scale, 20 * scale, 14 * scale);
    context.strokeRect(-10 * scale, -7 * scale, 20 * scale, 14 * scale);
    context.fillRect(-35 * scale, -5 * scale, 19 * scale, 10 * scale);
    context.strokeRect(-35 * scale, -5 * scale, 19 * scale, 10 * scale);
    context.fillRect(16 * scale, -5 * scale, 19 * scale, 10 * scale);
    context.strokeRect(16 * scale, -5 * scale, 19 * scale, 10 * scale);
    context.beginPath();
    context.moveTo(0, -7 * scale);
    context.lineTo(0, -18 * scale);
    context.moveTo(-5 * scale, -14 * scale);
    context.lineTo(5 * scale, -14 * scale);
    context.moveTo(-10 * scale, 0);
    context.lineTo(-16 * scale, 0);
    context.moveTo(10 * scale, 0);
    context.lineTo(16 * scale, 0);
    context.stroke();

    context.font = `${8 * scale}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    context.fillStyle = "rgba(216, 243, 255, 0.7)";
    context.textAlign = "center";
    context.fillText("DEPOT", 0, 28 * scale);
    context.restore();
  }

  function render(timestamp = 0) {
    const time = timestamp / 1000;
    const layout = geometry();
    context.clearRect(0, 0, width, height);

    layout.outerRings.forEach((ring, index) => drawOrbit(ring, layout, 0.105 - index * 0.022));
    outerConstellation.forEach((satellite) => {
      const point = pointOnOrbit(
        layout.outerRings[satellite.orbit],
        time * satellite.speed + satellite.phase,
        layout,
      );
      drawSatellite(point, satellite.scale, false, 0.42);
    });

    const outerDepotAngle = time * outerShellSpeeds[0] + 2.82;
    const outerDepot = pointOnOrbit(layout.outerRings[0], outerDepotAngle, layout);
    drawDepot(outerDepot, 0.55, 0.82);

    layout.rings.forEach((ring, index) => drawOrbit(ring, layout, 0.3 - index * 0.055));

    drawEarth(layout);

    constellation.forEach((satellite) => {
      const point = pointOnOrbit(
        layout.rings[satellite.orbit],
        time * satellite.speed + satellite.phase,
        layout,
      );
      drawSatellite(point, satellite.scale, false);
    });

    const depotAngle = time * shellSpeeds[1] + 5.05;
    const depot = pointOnOrbit(layout.rings[1], depotAngle, layout);
    drawDepot(depot);

    if (!reducedMotion.matches) {
      animationFrame = window.requestAnimationFrame(render);
    }
  }

  function restart() {
    window.cancelAnimationFrame(animationFrame);
    render(0);
  }

  resize();
  render(0);
  window.addEventListener("resize", () => {
    resize();
    if (reducedMotion.matches) render(0);
  });
  reducedMotion.addEventListener("change", restart);
})();
