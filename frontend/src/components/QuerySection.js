import React, { useState, useEffect } from 'react';
import { TextField, Button, Box, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import { queryDocument, checkDocumentStatus } from '../services/api';

const QuerySection = ({ file }) => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [docStatus, setDocStatus] = useState(null);

  // Add status checking
  useEffect(() => {
    if (file?.id) {
      const checkStatus = async () => {
        try {
          const status = await checkDocumentStatus(file.id);
          setDocStatus(status);
          if (status === "processing") {
            setTimeout(checkStatus, 2000); // Poll every 2 seconds
          }
        } catch (err) {
          console.error("Status check failed:", err);
        }
      };
      checkStatus();
    }
  }, [file]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (docStatus !== "completed") {
      setError("Document is still being processed. Please wait.");
      return;
    }
    setLoading(true);
    try {
      const result = await queryDocument(query, file?.id);
      setResponse(result);
      setError(null);
    } catch (err) {
      setError('Failed to get response');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ my: 2 }}>
      {docStatus === "processing" && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Document is being processed. Please wait...
        </Alert>
      )}
      <Typography variant="h6" gutterBottom>Ask Questions</Typography>
      <form onSubmit={handleSubmit}>
        <TextField
          fullWidth
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          margin="normal"
          label="Your Question"
          disabled={loading}
        />
        <Button 
          type="submit" 
          variant="contained" 
          disabled={loading || !query.trim()}
        >
          {loading ? <CircularProgress size={24} /> : 'Ask'}
        </Button>
      </form>

      {response && (
        <Paper elevation={3} sx={{ p: 2, mt: 2, backgroundColor: '#f5f5f5' }}>
          <Typography variant="h6" gutterBottom>Answer:</Typography>
          <Typography paragraph>{response.answer}</Typography>
          
          {response.source_text && (
            <>
              <Typography variant="subtitle2" color="textSecondary">
                Source Text:
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                {response.source_text}
              </Typography>
            </>
          )}
          
          <Typography variant="caption" color="textSecondary">
            Confidence: {(response.confidence * 100).toFixed(1)}%
          </Typography>
        </Paper>
      )}

      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}
    </Box>
  );
};

export default QuerySection;
