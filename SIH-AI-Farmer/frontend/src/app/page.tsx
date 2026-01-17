'use client';

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import InputForm from "@/components/InputForm";
import ResultDisplay from "@/components/ResultDisplay";
import DemoButton from "@/components/DemoButton";
import { motion } from "framer-motion";
import { supabase } from "@/lib/supabase";

export default function Home() {
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);
  const router = useRouter();

  useEffect(() => {
    const checkUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
      if (user) router.push("/dashboard");
    };
    checkUser();
  }, [router]);

  if (user) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-2xl mx-auto"
    >
      <h1 className="text-3xl font-bold mb-4 text-center bg-gradient-to-r from-blue-600 to-green-600 text-transparent bg-clip-text">
        AI Farmer Assistant
      </h1>
      <DemoButton setResult={setResult} setError={setError} />
      <InputForm setResult={setResult} setError={setError} />
      {error && <p className="text-red-500">{error}</p>}
      {result && <ResultDisplay result={result} />}
    </motion.div>
  );
}