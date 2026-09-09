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
  const shuttleMissions = [
    { targetOrbit: 0, targetPhase: (2 / 9) * Math.PI * 2, cycleOffset: 0.08, duration: 24 },
    { targetOrbit: 2, targetPhase: (7 / 13) * Math.PI * 2 + 0.62, cycleOffset: 0.57, duration: 30 },
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

  function drawDepot(point) {
    const scale = width < 720 ? 0.72 : 0.9;
    context.save();
    context.translate(point.x, point.y);
    context.rotate(-0.18);
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

  function shortestAngleDifference(start, end) {
    const fullTurn = Math.PI * 2;
    return ((((end - start + Math.PI) % fullTurn) + fullTurn) % fullTurn) - Math.PI;
  }

  function spiralPoint(startRing, endRing, startAngle, endAngle, progress, layout, turns = 2) {
    const direction = endRing[0] >= startRing[0] ? 1 : -1;
    const angularTravel = shortestAngleDifference(startAngle, endAngle) + direction * turns * Math.PI * 2;
    const angle = startAngle + angularTravel * progress;
    const radiusX = startRing[0] + (endRing[0] - startRing[0]) * progress;
    const radiusY = startRing[1] + (endRing[1] - startRing[1]) * progress;
    return pointOnOrbit([radiusX, radiusY], angle, layout);
  }

  function drawShuttle(point, heading, opacity) {
    const scale = width < 720 ? 0.62 : 0.78;
    context.save();
    context.translate(point.x, point.y);
    context.rotate(heading);
    context.globalAlpha = opacity;
    context.fillStyle = "rgba(33, 133, 193, 0.96)";
    context.strokeStyle = "rgba(96, 195, 246, 0.9)";
    context.lineWidth = 1;
    context.fillRect(-7 * scale, -4 * scale, 14 * scale, 8 * scale);
    context.strokeRect(-7 * scale, -4 * scale, 14 * scale, 8 * scale);
    context.fillStyle = "rgba(171, 226, 255, 0.88)";
    context.fillRect(-16 * scale, -3 * scale, 7 * scale, 6 * scale);
    context.fillRect(9 * scale, -3 * scale, 7 * scale, 6 * scale);
    context.beginPath();
    context.arc(7 * scale, 0, 2.4 * scale, -Math.PI / 2, Math.PI / 2);
    context.strokeStyle = "rgba(218, 246, 255, 0.92)";
    context.stroke();
    context.restore();
  }

  function drawRendezvousHold(target, shuttle) {
    context.save();
    context.beginPath();
    context.arc(target.x, target.y, 16, 0, Math.PI * 2);
    context.strokeStyle = "rgba(190, 235, 255, 0.34)";
    context.stroke();
    context.beginPath();
    context.moveTo(target.x, target.y);
    context.lineTo(shuttle.x, shuttle.y);
    context.strokeStyle = "rgba(211, 244, 255, 0.5)";
    context.setLineDash([2, 3]);
    context.stroke();
    context.restore();
  }

  function drawShuttleMissions(time, layout, depot, depotAngle) {
    shuttleMissions.forEach((mission) => {
      const targetAngle = time * shellSpeeds[mission.targetOrbit] + mission.targetPhase;
      const target = pointOnOrbit(
        layout.rings[mission.targetOrbit],
        targetAngle,
        layout,
      );
      const rawCycle = ((time / mission.duration + mission.cycleOffset) % 1 + 1) % 1;
      const outboundEnd = 0.35;
      const rendezvousEnd = 0.56;
      const returnEnd = 0.91;
      const outbound = rawCycle < outboundEnd;
      const rendezvous = rawCycle >= outboundEnd && rawCycle < rendezvousEnd;
      const returning = rawCycle >= rendezvousEnd && rawCycle < returnEnd;
      const legProgress = outbound
        ? rawCycle / outboundEnd
        : returning
          ? (rawCycle - rendezvousEnd) / (returnEnd - rendezvousEnd)
          : 0;
      const eased = legProgress * legProgress * (3 - 2 * legProgress);
      if (outbound || returning) {
        const startRing = outbound ? layout.rings[1] : layout.rings[mission.targetOrbit];
        const endRing = outbound ? layout.rings[mission.targetOrbit] : layout.rings[1];
        const startAngle = outbound ? depotAngle : targetAngle;
        const endAngle = outbound ? targetAngle : depotAngle;
        context.save();
        context.beginPath();
        for (let step = 0; step <= 64; step += 1) {
          const pathProgress = step / 64;
          const pathPoint = spiralPoint(
            startRing,
            endRing,
            startAngle,
            endAngle,
            pathProgress,
            layout,
          );
          if (step === 0) context.moveTo(pathPoint.x, pathPoint.y);
          else context.lineTo(pathPoint.x, pathPoint.y);
        }
        context.strokeStyle = "rgba(143, 218, 255, 0.13)";
        context.setLineDash([3, 7]);
        context.stroke();
        context.restore();

        const shuttle = spiralPoint(startRing, endRing, startAngle, endAngle, eased, layout);
        const behind = spiralPoint(
          startRing,
          endRing,
          startAngle,
          endAngle,
          Math.max(eased - 0.002, 0),
          layout,
        );
        const ahead = spiralPoint(
          startRing,
          endRing,
          startAngle,
          endAngle,
          Math.min(eased + 0.002, 1),
          layout,
        );
        const heading = Math.atan2(ahead.y - behind.y, ahead.x - behind.x);
        const endpointFade = Math.min(1, Math.sin(Math.PI * legProgress) * 2.8 + 0.2);
        drawShuttle(shuttle, heading, endpointFade);
      } else if (rendezvous) {
        const pulse = Math.sin(((rawCycle - outboundEnd) / (rendezvousEnd - outboundEnd)) * Math.PI * 2);
        const shuttle = { x: target.x + 13, y: target.y - 9 + pulse * 1.5 };
        drawRendezvousHold(target, shuttle);
        drawShuttle(shuttle, -0.18, 1);
      } else {
        const shuttle = { x: depot.x + 15, y: depot.y - 8 };
        drawShuttle(shuttle, -0.18, 0.72);
      }
    });
  }

  function drawRendezvous(time, layout) {
    const ring = layout.rings[0];
    const clientAngle = time * shellSpeeds[0] + 0.15;
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
    drawShuttleMissions(time, layout, depot, depotAngle);
    drawDepot(depot);

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
