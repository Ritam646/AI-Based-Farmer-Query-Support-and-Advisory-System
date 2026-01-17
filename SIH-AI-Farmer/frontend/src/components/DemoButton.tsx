'use client';

import { motion } from "framer-motion";
import { postQuery } from "@/lib/api";

interface DemoButtonProps {
  setResult: (result: any) => void;
  setError: (error: string | null) => void;
}

export default function DemoButton({ setResult, setError }: DemoButtonProps) {
  const handleDemo = async (inputType: "text" | "image" | "voice") => {
    setError(null);
    try {
      const formData = new FormData();
      formData.append("input_type", inputType);
      formData.append("tgt_lang", "hin_Deva");

      if (inputType === "text") {
        formData.append("text", "What is red rust in mango?");
      } else if (inputType === "image") {
        const response = await fetch("/images/demo-mango.jpg");
        const blob = await response.blob();
        formData.append("file", blob, "demo-mango.jpg");
      } else {
        const response = await fetch("/audio/demo-voice.wav");
        const blob = await response.blob();
        formData.append("file", blob, "demo-voice.wav");
      }

      const result = await postQuery(formData);
      setResult(result);
    } catch (err: any) {
      setError(err.message || "Demo failed");
    }
  };

  return (
    <div className="flex space-x-4 mb-4">
      <motion.button
        onClick={() => handleDemo("text")}
        className="bg-green-600 text-white p-2 rounded hover:bg-green-700"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        Demo Text
      </motion.button>
      <motion.button
        onClick={() => handleDemo("image")}
        className="bg-green-600 text-white p-2 rounded hover:bg-green-700"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        Demo Image
      </motion.button>
      <motion.button
        onClick={() => handleDemo("voice")}
        className="bg-green-600 text-white p-2 rounded hover:bg-green-700"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        Demo Voice
      </motion.button>
    </div>
  );
}