'use client';

import { useState } from "react";
import { useRouter } from "next/navigation";
import AuthForm from "@/components/AuthForm";
import { supabase } from "@/lib/supabase";
import toast from "react-hot-toast";

export default function SignUp() {
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSignUp = async (email: string, password: string) => {
    setLoading(true);
    try {
      const { error } = await supabase.auth.signUp({ email, password });
      if (error) throw error;
      toast.success("Signed up successfully");
      router.push("/dashboard");
    } catch (err: any) {
      toast.error(err.message || "Sign-up failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-10">
      <h1 className="text-3xl font-bold mb-4 text-center bg-gradient-to-r from-blue-600 to-green-600 text-transparent bg-clip-text">
        Sign Up
      </h1>
      <AuthForm type="sign-up" onSubmit={handleSignUp} loading={loading} />
    </div>
  );
}