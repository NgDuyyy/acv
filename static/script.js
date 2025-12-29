document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const resultLayout = document.getElementById('result-layout');
    const previewImg = document.getElementById('preview-img');
    const removeBtn = document.getElementById('remove-btn');
    const generateBtn = document.getElementById('generate-btn');
    const captionText = document.getElementById('caption-text');
    const copyBtn = document.getElementById('copy-btn');
    const loadingBar = document.getElementById('loading-bar');
    const statusText = document.getElementById('status-text');

    let currentFile = null;

    // Drag & Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadZone.classList.add('drag-active');
    }

    function unhighlight(e) {
        uploadZone.classList.remove('drag-active');
    }

    uploadZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
            } else {
                alert('Please upload an image file (JPG/PNG).');
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            previewImg.src = reader.result;
            uploadContent.style.display = 'none';
            resultLayout.classList.remove('hidden');
            resultLayout.style.display = 'grid'; // Ensure grid layout
            
            // Reset state
            captionText.textContent = '...';
            captionText.style.color = 'var(--text-secondary)';
            statusText.textContent = 'Ready to generate';
            generateBtn.disabled = false;
            loadingBar.classList.remove('visible');
             loadingBar.classList.remove('active');
        }
    }

    // Remove Image
    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        uploadContent.style.display = 'flex'; // Restore flex layout
        resultLayout.classList.add('hidden');
        resultLayout.style.display = 'none';
    });

    // Generate Caption
    generateBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Updates
        generateBtn.disabled = true;
        statusText.textContent = 'Analyzing image...';
        loadingBar.classList.remove('hidden');
        loadingBar.classList.add('visible'); // Show container
        loadingBar.classList.add('active'); // Start animation
        
        captionText.textContent = '...';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const data = await response.json();
            
            // Success
            statusText.textContent = 'Caption generated successfully!';
            captionText.textContent = data.caption;
            captionText.style.color = 'var(--text-primary)';
            
            // Typewriter effect simulation (optional improvement)
            
        } catch (error) {
            console.error('Error:', error);
            statusText.textContent = 'Error: ' + error.message;
            captionText.textContent = 'Failed to generate caption. Please try again.';
            captionText.style.color = '#ef4444'; // Red
        } finally {
            generateBtn.disabled = false;
            loadingBar.classList.remove('active');
            setTimeout(() => {
                 // loadingBar.classList.remove('visible');
            }, 1000);
        }
    });

    // Copy to Clipboard
    copyBtn.addEventListener('click', () => {
        const text = captionText.textContent;
        if (text && text !== '...') {
            navigator.clipboard.writeText(text).then(() => {
                const originalHtml = copyBtn.innerHTML;
                copyBtn.innerHTML = `<span style="color: var(--success)">✓ Copied!</span>`;
                setTimeout(() => {
                    copyBtn.innerHTML = originalHtml;
                }, 2000);
            });
        }
    });
});
