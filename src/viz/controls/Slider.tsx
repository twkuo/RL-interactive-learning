// Shared slider.
interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals?: number;
  onChange: (v: number) => void;
}

export function Slider({ label, value, min, max, step, decimals = 2, onChange }: Props) {
  return (
    <div className="slider">
      <div className="slider-head">
        <span>{label}</span>
        <span className="mono">{value.toFixed(decimals)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}
