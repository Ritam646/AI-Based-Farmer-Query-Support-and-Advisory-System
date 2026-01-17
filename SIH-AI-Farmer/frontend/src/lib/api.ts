import axios from "axios";
import { supabase } from "./supabase"; // Import Supabase client

const getAuthHeader = async () => {
  const { data: { session } } = await supabase.auth.getSession();
  return session?.access_token ? `Bearer ${session.access_token}` : undefined;
};

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
});

export const postQuery = async (formData: FormData) => {
  const authHeader = await getAuthHeader();
  const headers = { "Content-Type": "multipart/form-data" };
  if (authHeader) headers.Authorization = authHeader;

  const response = await api.post("/query", formData, { headers });
  return response.data;
};

export const postDisease = async (formData: FormData) => {
  const authHeader = await getAuthHeader();
  const headers = { "Content-Type": "multipart/form-data" };
  if (authHeader) headers.Authorization = authHeader;

  const response = await api.post("/disease/predict", formData, { headers });
  return response.data;
};

export const postRag = async (query: string) => {
  const authHeader = await getAuthHeader();
  const headers = authHeader ? { Authorization: authHeader } : {};
  const response = await api.post("/rag", { query }, { headers });
  return response.data;
};

export const postMarket = async (query: string) => {
  const authHeader = await getAuthHeader();
  const headers = { "Content-Type": "application/x-www-form-urlencoded", ...(authHeader ? { Authorization: authHeader } : {}) };
  const response = await api.post("/market/query", { query }, { headers });
  return response.data;
};

export const postTranscription = async (formData: FormData) => {
  const authHeader = await getAuthHeader();
  const headers = { "Content-Type": "multipart/form-data" };
  if (authHeader) headers.Authorization = authHeader;

  const response = await api.post("/transcription/transcribe", formData, { headers });
  return response.data;
};