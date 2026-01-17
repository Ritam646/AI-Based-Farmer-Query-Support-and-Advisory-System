'use client';

import { useState } from "react";
import { motion } from "framer-motion";
import { FaImage, FaMicrophone, FaTextHeight } from "react-icons/fa";
import { postQuery } from "@/lib/api";

interface InputFormProps {
  setResult: (result: any) => void;
  setError: (error: string | null) => void;
}

export default function InputForm({ setResult, setError }: InputFormProps) {
  const [inputType, setInputType] = useState<"text" | "image" | "voice">("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("hin_Deva");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("input_type", inputType);
      formData.append("tgt_lang", language);
      if (inputType === "text") {
        formData.append("text", text);
      } else {
        if (file) formData.append("file", file);
      }

      const response = await postQuery(formData);
      setResult(response);
    } catch (err: any) {
      setError(err.message || "Failed to process query");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.form
      onSubmit={handleSubmit}
      className="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg shadow-md"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Input Type</label>
        <div className="flex space-x-4">
          <button
            type="button"
            onClick={() => setInputType("text")}
            className={`p-2 rounded ${inputType === "text" ? "bg-blue-600 text-white" : "bg-gray-300"}`}
          >
            <FaTextHeight />
          </button>
          <button
            type="button"
            onClick={() => setInputType("image")}
            className={`p-2 rounded ${inputType === "image" ? "bg-blue-600 text-white" : "bg-gray-300"}`}
          >
            <FaImage />
          </button>
          <button
            type="button"
            onClick={() => setInputType("voice")}
            className={`p-2 rounded ${inputType === "voice" ? "bg-blue-600 text-white" : "bg-gray-300"}`}
          >
            <FaMicrophone />
          </button>
        </div>
      </div>

      {inputType === "text" ? (
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Query</label>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full p-2 border rounded dark:bg-gray-700 dark:text-white"
            placeholder="e.g., What is red rust in mango?"
          />
        </div>
      ) : (
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Upload {inputType === "image" ? "Image" : "Audio"}</label>
          <input
            type="file"
            accept={inputType === "image" ? "image/*" : "audio/*"}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full p-2 border rounded dark:bg-gray-700 dark:text-white"
          />
        </div>
      )}

      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Language</label>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="w-full p-2 border rounded dark:bg-gray-700 dark:text-white"
        >
          <option value="hin_Deva">Hindi</option>
          <option value="eng_Latn">English</option>
          <option value="tam_Taml">Tamil</option>
          <option value="tel_Telu">Telugu</option>
        </select>
      </div>

      <motion.button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700 disabled:bg-gray-500"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {loading ? "Processing..." : "Submit"}
      </motion.button>
    </motion.form>
  );
}