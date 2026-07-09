export default function HelpCard() {
  return (
    <section className="card help-card">
      <span className="help-icon" aria-hidden="true">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="9" />
          <path d="M9.3 9.3a2.7 2.7 0 0 1 5.2 1c0 1.8-2.3 1.8-2.5 3.2M12 17h.01" />
        </svg>
      </span>
      <div>
        <h2>¿Tienes dudas?</h2>
        <p>
          Si algo no queda claro o el resultado no coincide con lo que observas, consulta
          con el equipo de coordinación del albergue antes de tomar una decisión.
        </p>
      </div>
    </section>
  );
}
