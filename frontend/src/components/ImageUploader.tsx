import React, { useRef, useState } from 'react';
import { UploadCloud, AlertCircle } from 'lucide-react';

interface Props {
  onImageSelected: (file: File) => void;
  isLoading: boolean;
  selectedPreview: string | null;
}

export const ImageUploader: React.FC<Props> = ({
  onImageSelected,
  isLoading,
  selectedPreview,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndHandleFile = (file: File) => {
    setErrorMsg(null);
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setErrorMsg('Invalid file format. Please upload a JPEG, PNG, or WEBP image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg('Image size exceeds 10MB limit.');
      return;
    }
    onImageSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndHandleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            validateAndHandleFile(e.target.files[0]);
          }
        }}
      />

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-6 transition-all cursor-pointer flex flex-col items-center justify-center text-center group min-h-[260px] ${
          isDragOver
            ? 'border-teal-400 bg-teal-950/20 shadow-lg shadow-teal-500/10'
            : 'border-slate-700 bg-slate-900/50 hover:border-slate-500 hover:bg-slate-800/40'
        }`}
      >
        {selectedPreview ? (
          <div className="flex flex-col items-center gap-3">
            <div className="relative w-44 h-44 rounded-xl overflow-hidden border border-slate-700 shadow-md">
              <img
                src={selectedPreview}
                alt="Selected lesion"
                className="w-full h-full object-cover"
              />
            </div>
            <p className="text-xs text-slate-400">Click or drag a new image to replace</p>
          </div>
        ) : (
          <>
            <div className="w-14 h-14 rounded-2xl bg-teal-500/10 text-teal-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <UploadCloud className="w-7 h-7" />
            </div>
            <h3 className="font-semibold text-slate-200 text-sm mb-1">
              Upload Dermatoscopic Skin Lesion
            </h3>
            <p className="text-xs text-slate-400 max-w-xs mb-3">
              Drag and drop an image here, or click to browse from your device
            </p>
            <div className="flex items-center gap-2 text-[11px] text-slate-500">
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">JPEG</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">PNG</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">WEBP</span>
              <span>• Max 10MB</span>
            </div>
          </>
        )}
      </div>

      {errorMsg && (
        <div className="mt-3 flex items-center gap-2 text-rose-400 text-xs bg-rose-950/30 border border-rose-800/50 p-2.5 rounded-lg">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};
