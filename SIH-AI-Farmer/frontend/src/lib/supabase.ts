import { createClient } from "@supabase/supabase-js";

   const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
   const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

   export const supabase = createClient(supabaseUrl, supabaseAnonKey);

   export async function getUserHistory(userId: string) {
     const { data, error } = await supabase
       .from("user_history")
       .select("*")
       .eq("user_id", userId)
       .order("created_at", { ascending: false });
     if (error) throw new Error(error.message);
     return data;
   }