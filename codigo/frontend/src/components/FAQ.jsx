const FAQS = [
  {
    question: '¿Qué significa "prioridad alta"?',
    answer:
      'Indica que el paciente podría necesitar atención médica lo antes posible, según los datos ingresados. Siempre debe confirmarse con el criterio del personal médico.',
  },
  {
    question: '¿Qué hago si no conozco un dato exacto?',
    answer:
      'Ingresa el valor más cercano que tengas disponible, por ejemplo el último registrado. El resultado es una estimación, no un diagnóstico exacto.',
  },
  {
    question: '¿Puedo confiar en el resultado sin revisarlo?',
    answer:
      'No. Esta herramienta es un apoyo para el trabajo diario del albergue; no reemplaza el criterio del personal médico o administrativo.',
  },
  {
    question: '¿Quiénes realizaron este proyecto?',
    answer:
      'Este proyecto fue realizado por Diego Meléndez, Juan Sánchez, Cesar Alvarado, Raúl Gutiérrez y Marcos Bedia.',
  },
];

export default function FAQ() {
  return (
    <section className="card faq">
      <h2>Preguntas frecuentes</h2>
      {FAQS.map((item) => (
        <details className="faq-item" key={item.question}>
          <summary>{item.question}</summary>
          <p>{item.answer}</p>
        </details>
      ))}
    </section>
  );
}
