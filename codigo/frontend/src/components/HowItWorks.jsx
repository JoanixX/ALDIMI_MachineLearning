const STEPS = [
  {
    number: 1,
    title: 'Elige una tarea',
    description: 'Selecciona qué necesitas evaluar: un paciente, el inventario o las donaciones.',
  },
  {
    number: 2,
    title: 'Ingresa los datos',
    description: 'Completa el formulario con la información que tengas disponible.',
  },
  {
    number: 3,
    title: 'Obtén el resultado',
    description: 'Recibe una respuesta clara y lista para usar en tu trabajo diario.',
  },
];

export default function HowItWorks() {
  return (
    <section className="how-it-works">
      <div className="how-grid">
        {STEPS.map((step) => (
          <div className="how-step" key={step.number}>
            <span className="how-step-number">{step.number}</span>
            <div>
              <h3>{step.title}</h3>
              <p>{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
