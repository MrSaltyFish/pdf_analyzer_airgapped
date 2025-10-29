import React, { useState } from 'react';
import { Button, Box, Alert } from '@mui/material';
import { uploadDocument } from '../services/api';

const FileUpload = ({ onFileUpload }) => {
  const [error, setError] = useState(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      try {
        const response = await uploadDocument(file);
        onFileUpload(response);
      } catch (err) {
        setError('Failed to upload file');
        console.error(err);
      }
    }
  };

  return (
    <Box sx={{ my: 2 }}>
      {error && <Alert severity="error">{error}</Alert>}
      <Button variant="contained" component="label">
        Upload PDF
        <input
          type="file"
          hidden
          accept=".pdf"
          onChange={handleFileChange}
        />
      </Button>
    </Box>
  );
};

export default FileUpload;
