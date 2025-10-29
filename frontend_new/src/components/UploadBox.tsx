export default function UploadBox({
  onFileUpload,
}: {
  onFileUpload: (file: File) => void;
}) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="border-2 border-dashed border-gray-400 rounded-lg p-4 mb-4 text-center">
      <p className="mb-2">📄 Upload PDF to start a session</p>
      <input type="file" accept=".pdf" onChange={handleChange} />
    </div>
  );
}
