import { Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';

interface Props {
  name?: string;
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export class VizErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(
      `[VizErrorBoundary] ${this.props.name ?? 'Visualization'} failed to render:`,
      error,
      info.componentStack,
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          style={{
            padding: '2rem',
            border: '1px solid var(--color-border, #e5e7eb)',
            borderRadius: '0.5rem',
            textAlign: 'center',
            color: 'var(--color-text-muted, #6b7280)',
          }}
        >
          <p style={{ margin: 0 }}>
            This visualization couldn&apos;t load
            {this.props.name ? ` (${this.props.name})` : ''}.
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}
