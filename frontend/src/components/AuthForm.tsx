'use client';

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";

interface AuthFormProps {
  type: "sign-in" | "sign-up";
  onSubmit: (email: string, password: string) => void;
  loading: boolean;
}

export default function AuthForm({ type, onSubmit, loading }: AuthFormProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(email, password);
  };

  return (
    <motion.form
      onSubmit={handleSubmit}
      className="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg shadow-md"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Email</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full p-2 border rounded dark:bg-gray-700 dark:text-white"
          required
        />
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full p-2 border rounded dark:bg-gray-700 dark:text-white"
          required
        />
      </div>
      <motion.button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700 disabled:bg-gray-500"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {loading ? "Processing..." : type === "sign-in" ? "Sign In" : "Sign Up"}
      </motion.button>
      <p className="mt-4 text-center">
        {type === "sign-in" ? (
          <>
            Don't have an account?{" "}
            <Link href="/auth/sign-up" className="text-blue-600 hover:underline">
              Sign Up
            </Link>
          </>
        ) : (
          <>
            Already have an account?{" "}
            <Link href="/auth/sign-in" className="text-blue-600 hover:underline">
              Sign In
            </Link>
          </>
        )}
      </p>
    </motion.form>
  );
}