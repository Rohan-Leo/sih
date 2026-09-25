export function buildTimeline(
  coords: [number, number][],
  opts?: { cruise?: number; turn?: number; accel?: number; dtMs?: number },
): { lat: number; lon: number; t: number }[]
