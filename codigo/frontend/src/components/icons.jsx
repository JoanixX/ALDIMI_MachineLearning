// Iconos de trazo simple (sin emojis) para la navegación por tarea.
// Heredan el color del texto via currentColor.
const common = {
  width: 22,
  height: 22,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
};

function PatientIcon() {
  return (
    <svg {...common}>
      <circle cx="12" cy="7.5" r="3.25" />
      <path d="M5 20.5c0-3.6 3.13-6 7-6s7 2.4 7 6" />
      <path d="M12 11v3.2M10.3 12.6h3.4" />
    </svg>
  );
}

function InventoryIcon() {
  return (
    <svg {...common}>
      <path d="M3.5 8.2 12 4l8.5 4.2v8.6L12 21l-8.5-4.2z" />
      <path d="M3.5 8.2 12 12l8.5-4.2M12 12v9" />
    </svg>
  );
}

function DonationsIcon() {
  return (
    <svg {...common}>
      <path d="M12 20.2s-7.4-4.4-7.4-9.7A4.1 4.1 0 0 1 12 7.9a4.1 4.1 0 0 1 7.4 2.6c0 5.3-7.4 9.7-7.4 9.7z" />
    </svg>
  );
}

export const TASK_ICONS = {
  patient: PatientIcon,
  inventory: InventoryIcon,
  donations: DonationsIcon,
};
