'use client';

import Link from "next/link";
import { useState, useEffect } from "react";
import { supabase } from "@/lib/supabase";

export default function Header() {
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    const checkUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
    };
    checkUser();
  }, []);

  return (
    <header className="gradient-bg text-white p-4">
      <nav className="container mx-auto flex justify-between items-center">
        <h1 className="text-xl font-bold">AI Farmer</h1>
        <div className="space-x-4">
          <Link href="/" className="hover:underline">Home</Link>
          <Link href="/dashboard" className="hover:underline">Dashboard</Link>
          <Link href="/disease" className="hover:underline">Disease Detection</Link>
          <Link href="/rag" className="hover:underline">Knowledge Base</Link>
          <Link href="/market" className="hover:underline">Market Prices</Link>
          <Link href="/transcription" className="hover:underline">Voice Query</Link>
          {user ? (
            <Link href="/dashboard" className="hover:underline">Profile</Link>
          ) : (
            <>
              <Link href="/auth/sign-in" className="hover:underline">Sign In</Link>
              <Link href="/auth/sign-up" className="hover:underline">Sign Up</Link>
            </>
          )}
        </div>
      </nav>
    </header>
  );
}