'use client';

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { supabase, getUserHistory } from "@/lib/supabase";
import DashboardCard from "@/components/DashboardCard";
import toast from "react-hot-toast";

export default function Dashboard() {
  const [user, setUser] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const router = useRouter();

  useEffect(() => {
    const checkUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push("/auth/sign-in");
        return;
      }
      setUser(user);
      try {
        const historyData = await getUserHistory(user.id);
        setHistory(historyData);
      } catch (err: any) {
        toast.error("Failed to load history");
      }
    };
    checkUser();
  }, [router]);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.push("/auth/sign-in");
    toast.success("Signed out successfully");
  };

  if (!user) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-4xl mx-auto"
    >
      <h1 className="text-3xl font-bold mb-4 text-center bg-gradient-to-r from-blue-600 to-green-600 text-transparent bg-clip-text">
        Welcome, {user.email}
      </h1>
      <button
        onClick={handleSignOut}
        className="mb-4 bg-red-600 text-white p-2 rounded hover:bg-red-700"
      >
        Sign Out
      </button>
      <h2 className="text-2xl font-semibold mb-4">Your Query History</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {history.length > 0 ? (
          history.map((item) => (
            <div key={item.id} className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg shadow-md">
              <p><strong>Query:</strong> {item.input}</p>
              <p><strong>Response:</strong> {item.response}</p>
              <p><strong>Type:</strong> {item.input_type}</p>
              <p><strong>Date:</strong> {new Date(item.created_at).toLocaleString()}</p>
            </div>
          ))
        ) : (
          <p>No history yet. Try a query!</p>
        )}
      </div>
      <h2 className="text-2xl font-semibold mb-4">Explore Features</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <DashboardCard
          title="Disease Detection"
          description="Upload a leaf image to detect diseases."
          link="/disease"
        />
        <DashboardCard
          title="Knowledge Base"
          description="Ask questions about farming techniques."
          link="/rag"
        />
        <DashboardCard
          title="Market Prices"
          description="Get real-time commodity prices."
          link="/market"
        />
        <DashboardCard
          title="Voice Query"
          description="Use voice commands for assistance."
          link="/transcription"
        />
      </div>
    </motion.div>
  );
}