import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

// Note: StrictMode intentionally removed to prevent double-invocation
// of initialise() which loads heavy ML model CDN scripts.
createRoot(document.getElementById("root")!).render(<App />);
