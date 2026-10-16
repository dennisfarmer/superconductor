import { createRoot } from "react-dom/client";
import { ErrorBoundary } from "react-error-boundary";
import type { FallbackProps } from "react-error-boundary";
import "./atlas.css";
import { App } from "./App";

function Fallback({ error }: FallbackProps) {
  return (
    <div role="alert" style={{ padding: 24 }}>
      <p>Something went wrong:</p>
      <pre>{String(error?.message ?? error)}</pre>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <ErrorBoundary FallbackComponent={Fallback}>
    <App />
  </ErrorBoundary>
);
