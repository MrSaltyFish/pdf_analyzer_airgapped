import React, { useState } from 'react';
import { Container, Box, Typography } from '@mui/material';
import FileUpload from './components/FileUpload';
import QuerySection from './components/QuerySection';

function App() {
  const [currentFile, setCurrentFile] = useState(null);

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          PDF Analyzer
        </Typography>
        <FileUpload onFileUpload={setCurrentFile} />
        {currentFile && <QuerySection file={currentFile} />}
      </Box>
    </Container>
  );
}

export default App;
