'use client';

import { useState } from "react";
import InputForm from "@/components/InputForm";
import ResultDisplay from "@/components/ResultDisplay";
import { motion } from "framer-motion";

export default function RagPage() {
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-2xl mx-auto"
    >
      <h1 className="text-3xl font-bold mb-4 text-center bg-gradient-to-r from-blue-600 to-green-600 text-transparent bg-clip-text">
        Knowledge Base
      </h1>
      <InputForm setResult={setResult} setError={setError} />
      {error && <p className="text-red-500">{error}</p>}
      {result && <ResultDisplay result={result} />}
    </motion.div>
  );
}