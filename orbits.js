(() => {
  const canvas = document.querySelector("#orbital-field");
  const context = canvas.getContext("2d");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let animationFrame = 0;

  const satellites = [
    { orbit: 0, phase: 0.15, speed: 0.075, scale: 1 },
    { orbit: 1, phase: 2.35, speed: -0.048, scale: 0.84 },
    { orbit: 2, phase: 4.4, speed: 0.034, scale: 0.76 },
  ];

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

  function drawSatellite(point, scale = 1, active = false) {
    context.save();
    context.translate(point.x, point.y);
    context.rotate(-0.18);
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

  function drawRendezvous(time, layout) {
    const ring = layout.rings[0];
    const clientAngle = time * 0.075 + satellites[0].phase;
    const cycle = (time * 0.035) % (Math.PI * 2);
    const separation = 0.7 * (0.5 + 0.5 * Math.cos(cycle));
    const servicerAngle = clientAngle - separation;
    const client = pointOnOrbit(ring, clientAngle, layout);
    const servicer = pointOnOrbit(ring, servicerAngle, layout);

    context.beginPath();
    context.moveTo(servicer.x, servicer.y);
    context.lineTo(client.x, client.y);
    context.strokeStyle = `rgba(188, 234, 255, ${0.18 + (1 - separation / 0.7) * 0.34})`;
    context.setLineDash([2, 5]);
    context.stroke();
    context.setLineDash([]);

    drawSatellite(client, 1, false);
    drawSatellite(servicer, 0.72, true);

    if (separation < 0.08) {
      context.beginPath();
      context.arc(client.x, client.y, 20, 0, Math.PI * 2);
      context.strokeStyle = "rgba(188, 234, 255, 0.34)";
      context.stroke();
    }
  }

  function render(timestamp = 0) {
    const time = timestamp / 1000;
    const layout = geometry();
    context.clearRect(0, 0, width, height);

    layout.rings.forEach((ring, index) => drawOrbit(ring, layout, 0.3 - index * 0.055));

    satellites.slice(1).forEach((satellite) => {
      const point = pointOnOrbit(
        layout.rings[satellite.orbit],
        time * satellite.speed + satellite.phase,
        layout,
      );
      drawSatellite(point, satellite.scale, false);
    });

    drawRendezvous(time, layout);

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
