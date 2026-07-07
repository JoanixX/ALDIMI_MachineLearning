import { TASK_ICONS } from './icons.jsx';

export default function TaskMenu({ tasks, onSelect }) {
  return (
    <section className="task-menu">
      <h2>¿Qué necesitas hacer?</h2>
      <div className="task-grid">
        {tasks.map((task) => {
          const Icon = TASK_ICONS[task.icon];
          return (
            <button
              key={task.key}
              type="button"
              className={`task-card accent-${task.accent}`}
              onClick={() => onSelect(task.key)}
            >
              <span className="task-icon-badge">{Icon && <Icon />}</span>
              <span className="task-title">{task.title}</span>
              <span className="task-description">{task.description}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
