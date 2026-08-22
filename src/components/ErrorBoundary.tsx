import { Component, type ErrorInfo, type ReactNode } from "react";
import { CircleAlert } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
  label: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[${this.props.label} render error]`, error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <section className="panel error-boundary-card" role="alert">
        <div className="icon-tile">
          <CircleAlert aria-hidden="true" />
        </div>
        <div>
          <h2>{this.props.label} failed to render</h2>
          <p>{this.state.error.message}</p>
        </div>
      </section>
    );
  }
}
