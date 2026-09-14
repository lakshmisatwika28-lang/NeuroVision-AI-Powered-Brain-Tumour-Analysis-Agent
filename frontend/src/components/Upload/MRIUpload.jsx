import { useCallback, useRef, useState } from 'react';
import { UploadCloud, X, ImageIcon, Sparkles } from 'lucide-react';
import './MRIUpload.css';

const ACCEPTED = ['image/jpeg', 'image/jpg', 'image/png'];

export default function MRIUpload({ onAnalyze }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dims, setDims] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      setError('Please upload a JPG, JPEG, or PNG image.');
      return;
    }
    setError(null);
    const url = URL.createObjectURL(f);
    setFile(f);
    setPreviewUrl(url);
    const img = new Image();
    img.onload = () => setDims({ width: img.naturalWidth, height: img.naturalHeight });
    img.src = url;
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    handleFile(f);
  };

  const removeFile = () => {
    setFile(null);
    setPreviewUrl(null);
    setDims(null);
    setError(null);
  };

  return (
    <div className="mri-upload">
      {!file && (
        <div
          className={`mri-upload__dropzone ${dragging ? 'mri-upload__dropzone--active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
        >
          <div className="mri-upload__icon">
            <UploadCloud size={30} />
          </div>
          <h3>Upload Brain MRI</h3>
          <p>Drag &amp; drop your MRI here, or</p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
          >
            Browse Files
          </button>
          <span className="mri-upload__formats">JPG · JPEG · PNG</span>
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            hidden
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>
      )}

      {error && <p className="mri-upload__error">{error}</p>}

      {file && (
        <div className="mri-upload__preview fade-in">
          <div className="mri-upload__preview-image">
            <img src={previewUrl} alt="Uploaded MRI preview" />
            <button className="mri-upload__remove" onClick={removeFile} aria-label="Remove image">
              <X size={15} />
            </button>
          </div>
          <div className="mri-upload__meta">
            <div className="mri-upload__meta-row">
              <ImageIcon size={14} />
              <span className="mri-upload__filename" title={file.name}>{file.name}</span>
            </div>
            {dims && (
              <span className="mri-upload__dims mono">{dims.width} × {dims.height}px</span>
            )}
          </div>
          <button className="btn btn-primary mri-upload__analyze" onClick={() => onAnalyze(file)}>
            <Sparkles size={15} />
            Analyze MRI
          </button>
        </div>
      )}
    </div>
  );
}
