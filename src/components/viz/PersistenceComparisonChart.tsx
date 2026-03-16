import { vrIntervals, cechIntervals } from '../../data/cech-persistence-data';
import PersistenceDiagram from './PersistenceDiagram';

export default function PersistenceComparisonChart() {
  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
      <div>
        <h3
          className="mb-2 text-center text-sm font-semibold"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
        >
          Vietoris-Rips (Ripser)
        </h3>
        <p
          className="mb-3 text-center text-xs"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)' }}
        >
          {vrIntervals.length} intervals &middot; note phantom H<sub>1</sub> features
        </p>
        <PersistenceDiagram intervals={vrIntervals} mode="diagram" showDiagonal />
      </div>

      <div>
        <h3
          className="mb-2 text-center text-sm font-semibold"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
        >
          &#x10C;ech / Alpha (GUDHI)
        </h3>
        <p
          className="mb-3 text-center text-xs"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)' }}
        >
          {cechIntervals.length} intervals &middot; cleaner H<sub>1</sub> detection
        </p>
        <PersistenceDiagram intervals={cechIntervals} mode="diagram" showDiagonal />
      </div>
    </div>
  );
}
