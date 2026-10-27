import { LEAGUE_AVERAGE, formatCell } from "./format";

const WIDTH = 192;
const HEIGHT = 22;

interface Props {
  lo: number;
  hi: number;
  point: number | null;
  domain: [number, number];
}

/** One lineup's 95% interval against the league-average line: is it really different? */
export function IntervalStrip({ lo, hi, point, domain }: Props) {
  const [min, max] = domain;
  const x = (v: number) => ((v - min) / (max - min)) * WIDTH;
  const excludes = lo > LEAGUE_AVERAGE || hi < LEAGUE_AVERAGE;
  const label =
    `95% interval ${formatCell(lo, "")} to ${formatCell(hi, "")}` +
    (excludes ? ", excludes the league average" : ", includes the league average");
  return (
    <svg className="strip" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={label}>
      <line className="league" x1={x(LEAGUE_AVERAGE)} x2={x(LEAGUE_AVERAGE)} y1={0} y2={HEIGHT} />
      <rect
        className="band"
        x={x(lo)}
        y={7}
        width={Math.max(x(hi) - x(lo), 1)}
        height={8}
        rx={4}
      />
      {point !== null && <circle className="point" cx={x(point)} cy={11} r={3.5} />}
    </svg>
  );
}
