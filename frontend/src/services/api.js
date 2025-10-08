import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const queryDocument = async (question, context = null) => {
    try {
        const response = await api.post('/query/', { 
            question,
            context 
        });
        return response.data;
    } catch (error) {
        const errorMessage = error.response?.data?.detail || 'Failed to process query';
        console.error('Query error:', errorMessage);
        throw new Error(errorMessage);
    }
};

export const checkDocumentStatus = async (docId) => {
    try {
        const response = await api.get(`/docs/status/${docId}`);
        return response.data.status;
    } catch (error) {
        console.error('Status check error:', error);
        throw new Error(error.response?.data?.detail || 'Failed to check document status');
    }
};

export const uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await api.post('/docs/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Failed to upload document');
    }
};

export default api;
