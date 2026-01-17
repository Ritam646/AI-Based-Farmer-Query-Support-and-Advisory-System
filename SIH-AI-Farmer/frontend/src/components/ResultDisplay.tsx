'use client';

import { motion } from "framer-motion";

interface ResultDisplayProps {
  result: any;
}

export default function ResultDisplay({ result }: ResultDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mt-6 p-4 bg-white dark:bg-gray-700 rounded-lg shadow-md"
    >
      {result.predicted_disease_en ? (
        <>
          <h2 className="text-xl font-bold">Disease Detection</h2>
          <p><strong>English:</strong> {result.predicted_disease_en}</p>
          <p><strong>Translated:</strong> {result.predicted_disease_translated}</p>
          <p><strong>Language:</strong> {result.target_language}</p>
        </>
      ) : result.answer ? (
        <>
          <h2 className="text-xl font-bold">Knowledge Base</h2>
          <p>{result.answer}</p>
          {result.sources && (
            <ul className="list-disc pl-5">
              {result.sources.map((source: string, index: number) => (
                <li key={index}>{source}</li>
              ))}
            </ul>
          )}
        </>
      ) : result.response ? (
        <>
          <h2 className="text-xl font-bold">Market Analysis</h2>
          <pre className="whitespace-pre-wrap">{result.response}</pre>
        </>
      ) : result.transcription ? (
        <>
          <h2 className="text-xl font-bold">Transcription</h2>
          <p>{result.transcription}</p>
        </>
      ) : (
        <pre className="whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
      )}
    </motion.div>
  );
}