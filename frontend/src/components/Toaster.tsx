'use client';

import { Toaster as ReactToaster } from "react-hot-toast";

export default function Toaster() {
  return (
    <ReactToaster
      position="top-right"
      toastOptions={{
        style: {
          background: "var(--background)",
          color: "var(--foreground)",
        },
      }}
    />
  );
}