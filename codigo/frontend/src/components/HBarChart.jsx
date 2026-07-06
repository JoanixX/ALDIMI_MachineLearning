// Gráfico de barras horizontales de una sola serie (azul), con etiqueta
// directa por barra y tooltip nativo al pasar el cursor. Al ser una única
// serie no lleva leyenda: el título del bloque la nombra.
export default function HBarChart({ items, max = 1, format }) {
  const fmt = format || ((v) => v.toFixed(3));
  return (
    <div role="img" aria-label="Gráfico de barras">
      {items.map(({ label, value }) => (
        <div className="hbar-row" key={label} title={`${label}: ${fmt(value)}`}>
          <span className="hbar-label">{label}</span>
          <div className="hbar-track">
            <div
              className="hbar-fill"
              style={{ width: `${Math.max(0, Math.min(100, (value / max) * 100))}%` }}
            />
          </div>
          <span className="hbar-value">{fmt(value)}</span>
        </div>
      ))}
    </div>
  );
}
