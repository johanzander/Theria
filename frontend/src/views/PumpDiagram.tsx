import { fetchPumpNow } from '../api'
import { Panel, useLoaded } from '../ui'

const HOT = '#ff9830'
const COLD = '#8ab8ff'
const IDLE = '#555b63'
const PIPE = { fill: 'none', strokeWidth: 3, strokeLinejoin: 'round', strokeLinecap: 'round' } as const

const reading = (value: number | null, unit: string, digits = 1) => (value === null ? '-' : `${value.toFixed(digits)} ${unit}`)

function Value({ x, y, children, anchor = 'start' }: { x: number; y: number; children: string; anchor?: 'start' | 'middle' | 'end' }) {
  return <text x={x} y={y} textAnchor={anchor} fill="#fff" fontSize={17}>{children}</text>
}

function Exchanger({ x }: { x: number }) {
  return (
    <g>
      <rect x={x} y={190} width={30} height={120} rx={8} fill="#f4f4f4" stroke="#999" />
      {[8, 15, 22].map((dx) => <line key={dx} x1={x + dx} x2={x + dx} y1={196} y2={304} stroke="#777" />)}
    </g>
  )
}

/** The heat pump as a flow picture: which way the heat goes now and how warm each part is. */
export function PumpDiagram() {
  const { data } = useLoaded(fetchPumpNow)
  if (!data) return null
  const pump = data.values
  const hotWater = data.hot_water
  const running = (pump.comp ?? 0) > 0
  const heater = (pump.aux ?? 0) > 0
  const toRadiators = hotWater ? IDLE : HOT
  const toTank = hotWater ? HOT : IDLE
  return (
    <Panel
      title={`Heat pump now: ${hotWater ? 'hot water' : 'heating'}`}
      description="Where the heat goes right now. Orange pipes carry the heat out, blue ones bring the water back; grey is the way the valve is closed. All charts are on the Värmepump tab."
    >
      <svg viewBox="0 0 720 340" role="img" aria-label="Heat pump flow diagram" style={{ width: '100%', maxWidth: 720, display: 'block', margin: '0 auto', background: '#262626', borderRadius: 14 }}>
        <Value x={18} y={32}>{reading(pump.outdoor, '°C')}</Value>
        <path d="M 90 110 L 170 58 L 250 110 M 108 100 L 108 158 L 232 158 L 232 100" stroke="#fff" strokeWidth={4} fill="none" strokeLinejoin="round" />

        <path d="M 75 205 C 30 205 30 225 55 225 C 30 225 30 250 55 250 C 30 250 30 272 55 272 C 30 272 30 295 75 295" stroke={COLD} {...PIPE} />
        <path d="M 75 205 L 175 205" stroke="#e8a598" {...PIPE} />
        <path d="M 75 295 L 175 295" stroke={COLD} {...PIPE} />
        <Value x={82} y={197}>{reading(pump.evap, '°C')}</Value>
        <Value x={82} y={287}>{reading(pump.cond, '°C')}</Value>

        <path d="M 205 205 L 395 205" stroke={running ? HOT : IDLE} {...PIPE} />
        <path d="M 205 295 L 288 295 M 312 295 L 395 295" stroke={running ? COLD : IDLE} {...PIPE} />
        <path d="M 288 285 L 312 305 L 312 285 L 288 305 Z" fill="none" stroke="#ddd" strokeWidth={1.5} />
        <Value x={232} y={197}>{reading(pump.hotgas, '°C')}</Value>
        <path d="M 268 228 L 256 252 L 266 252 L 258 274 L 280 246 L 269 246 L 278 228 Z" fill={running ? '#fade2a' : IDLE} />
        <Value x={286} y={248}>{reading(pump.power, 'kW')}</Value>
        <Value x={286} y={272}>{reading(pump.comp, '%', 0)}</Value>

        <Exchanger x={175} />
        <Exchanger x={395} />

        <path d="M 425 205 L 469 205 L 469 170" stroke={HOT} {...PIPE} />
        <rect x={460} y={118} width={18} height={52} fill={heater ? '#f2495c' : '#7a3b42'} stroke="#222" />
        <path d="M 464 128 L 474 136 L 464 146 L 474 156" stroke="#222" fill="none" />
        <Value x={454} y={148} anchor="end">{reading(pump.aux, '%', 0)}</Value>
        <path d="M 469 118 L 469 88 L 520 88" stroke={HOT} {...PIPE} />

        <circle cx={528} cy={88} r={8} fill={hotWater ? '#fade2a' : '#ddd'} />
        <path d="M 520 88 L 410 88" stroke={toRadiators} {...PIPE} />
        <path d="M 536 88 L 600 88 L 600 122" stroke={toTank} {...PIPE} />

        <rect x={338} y={30} width={72} height={66} rx={8} fill="#f4f4f4" stroke="#999" />
        {[350, 362, 374, 386, 398].map((x) => <line key={x} x1={x} x2={x} y1={40} y2={86} stroke="#555" />)}
        <Value x={420} y={78}>{reading(pump.flow, '°C')}</Value>

        <ellipse cx={615} cy={126} rx={48} ry={13} fill="#d9823b" />
        <rect x={567} y={126} width={96} height={142} fill="#d9823b" />
        <ellipse cx={615} cy={268} rx={48} ry={13} fill="#b96a2c" />
        <Value x={615} y={170} anchor="middle">{reading(pump.dhwtop, '°C')}</Value>
        <Value x={615} y={228} anchor="middle">{reading(pump.dhwmid, '°C')}</Value>

        <path d="M 410 42 L 695 42 L 695 295 L 425 295" stroke={COLD} {...PIPE} />
        <path d="M 615 281 L 615 295" stroke={hotWater ? COLD : IDLE} {...PIPE} />
        <Value x={435} y={232}>{reading(pump.hc_fwd, '°C')}</Value>
        <Value x={435} y={285}>{reading(pump.hc_ret, '°C')}</Value>
      </svg>
    </Panel>
  )
}
