import { useState } from 'react';
import { FileText, Loader2 } from 'lucide-react';
import { generateReport } from '../../api/neurovisionApi';

export default function ReportButton({ analysisId }) {
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState(null);

  const handleClick = async () => {
    if (!analysisId) {
      setStatus('error');
      setMessage('Analysis ID is missing. Please analyze the MRI again.');
      return;
    }

    setStatus('loading');
    setMessage(null);

    try {
      const blob = await generateReport(analysisId);

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');

      link.href = url;
      link.download = 'NeuroVision_Analysis_Report.pdf';

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);

      setStatus('idle');
    } catch (err) {
      console.error('Report generation error:', err);
      setStatus('error');
      setMessage(err.message || 'Report generation failed.');
    }
  };

  return (
    <div>
      <button
        className="btn"
        onClick={handleClick}
        disabled={status === 'loading'}
      >
        {status === 'loading' ? (
          <Loader2 size={15} className="spin" />
        ) : (
          <FileText size={15} />
        )}

        {status === 'loading'
          ? 'Generating Report...'
          : 'Generate Analysis Report'}
      </button>

      {message && (
        <p className="report-btn__note">
          {message}
        </p>
      )}
    </div>
  );
}