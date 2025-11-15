document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Upload Section
    const qaForm = document.getElementById("qaForm");
    const fileInput = document.getElementById("qaPdfFile");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const fileInfo = document.getElementById("fileInfo");
    const uploadArea = document.getElementById("uploadArea");
    const removeFileBtn = document.getElementById("removeFile");
    const professionSelect = document.getElementById("profession");
    const questionCountSlider = document.getElementById("questionCount");
    const questionCountValue = document.getElementById("questionCountValue");
    const mcqCount = document.getElementById("mcqCount");
    const shortCount = document.getElementById("shortCount");
    const generateBtn = document.getElementById("generateBtn");

    // DOM Elements - Sections
    const uploadSection = document.getElementById("uploadSection");
    const processingProgress = document.getElementById("processingProgress");
    const quizSection = document.getElementById("quizSection");

    // DOM Elements - Progress Tracking
    const progressFill = document.getElementById("progressFill");
    const progressPercentage = document.getElementById("progressPercentage");
    const progressStatus = document.getElementById("progressStatus");
    const pageCount = document.getElementById("pageCount");
    const chunkCount = document.getElementById("chunkCount");
    const tokenUsage = document.getElementById("tokenUsage");
    const questionsGenerated = document.getElementById("questionsGenerated");
    const completionSection = document.getElementById("completionSection");

    // DOM Elements - Quiz Pages
    const mcqPage = document.getElementById("mcqPage");
    const onelinePage = document.getElementById("onelinePage");
    const step1 = document.getElementById("step1");
    const step2 = document.getElementById("step2");

    // DOM Elements - Navigation
    const mcqNextBtn = document.getElementById("mcqNextBtn");
    const onelinePrevBtn = document.getElementById("onelinePrevBtn");
    const submitBtn = document.getElementById("submitBtn");
    const backToUploadBtn = document.getElementById("backToUploadBtn");
    const exportResultsBtn = document.getElementById("exportResultsBtn");
    const retakeQuizBtn = document.getElementById("retakeQuizBtn");
    const actionButtons = document.getElementById("actionButtons");

    // DOM Elements - Statistics
    const mcqAnswered = document.getElementById("mcqAnswered");
    const mcqTotal = document.getElementById("mcqTotal");
    const onelineAnswered = document.getElementById("onelineAnswered");
    const onelineTotal = document.getElementById("onelineTotal");
    const finalSummary = document.getElementById("finalSummary");

    // Global Variables
    let MCQ_ITEMS = [];
    let ONE_LINE_ITEMS = [];
    let currentAnswers = { mcq: {}, oneline: {} };
    let evaluationResults = null;
    let progressInterval = null;
    let sessionData = {};
    let currentPage = 'upload';
    // ==================== TOKEN USAGE DEBUG BOX ==================== 

    let tokenUsageData = {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        estimatedCost: 0,
        currentStage: 'idle',
        isVisible: false,
        progressPercentage: 0
    };

    // Token Usage Functions
    function setupTokenUsageCloseButton() {
        const closeBtn = document.getElementById('tokenClose');
        const container = document.getElementById('tokenUsageContainer');
        
        if (closeBtn && container) {
            closeBtn.addEventListener('click', () => {
                toggleTokenUsageSlide();
            });
            
            container.addEventListener('click', (e) => {
                if (container.classList.contains('minimized') && e.target === container) {
                    toggleTokenUsageSlide();
                }
            });
        }
    }

    function toggleTokenUsageSlide() {
        const container = document.getElementById('tokenUsageContainer');
        const closeBtn = document.getElementById('tokenClose');
        
        if (!container) return;
        
        if (container.classList.contains('minimized')) {
            // Slide back in
            container.classList.remove('minimized');
            container.style.transform = 'translateX(0)';
            if (closeBtn) {
                closeBtn.innerHTML = '×';
                closeBtn.title = 'Minimize';
            }
        } else {
            // Slide out (minimize)
            container.classList.add('minimized');
            container.style.transform = 'translateX(calc(100% - 60px))';
            if (closeBtn) {
                closeBtn.innerHTML = '«';
                closeBtn.title = 'Expand';
            }
        }
    }

    function showTokenUsage(stage = 'generating') {
        const container = document.getElementById('tokenUsageContainer');
        const statusElement = document.getElementById('tokenStatus');
        
        if (container) {
            container.classList.remove('hidden', 'minimized');
            container.style.transform = 'translateX(0)';
            container.style.opacity = '1';
            tokenUsageData.isVisible = true;
            tokenUsageData.currentStage = stage;
            
            if (statusElement) {
                const statusMessages = {
                    'generating': '🔄 Generating questions...',
                    'evaluating': '📊 Evaluating answers...',
                    'complete': '✅ Process complete!'
                };
                statusElement.textContent = statusMessages[stage] || '⏳ Processing...';
            }
            
            container.className = `token-usage-container ${stage}`;
            
            const closeBtn = document.getElementById('tokenClose');
            if (closeBtn) {
                closeBtn.innerHTML = '×';
                closeBtn.title = 'Minimize';
                closeBtn.style.opacity = '1';
                closeBtn.style.pointerEvents = 'auto';
            }
        }
    }

    function updateTokenDisplay(inputTokens, outputTokens, estimatedCost = null, stage = 'generating') {
        const totalTokens = inputTokens + outputTokens;
        
        tokenUsageData = {
            ...tokenUsageData,
            inputTokens,
            outputTokens,
            totalTokens,
            estimatedCost: estimatedCost || calculateEstimatedCost(totalTokens),
            currentStage: stage
        };
        
        updateTokenValue('inputTokens', inputTokens.toLocaleString());
        updateTokenValue('outputTokens', outputTokens.toLocaleString());
        updateTokenValue('totalTokens', totalTokens.toLocaleString());
        updateTokenValue('estimatedCost', `$${tokenUsageData.estimatedCost.toFixed(4)}`);
        
        showTokenUsage(stage);
        updateTokenProgress(stage);
        
        console.log(`🔢 Token Usage - Input: ${inputTokens}, Output: ${outputTokens}, Total: ${totalTokens}`);
    }

    function updateTokenValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add('updating');
            setTimeout(() => {
                element.textContent = value;
                element.classList.remove('updating');
            }, 300);
        }
    }

    function updateTokenProgress(stage) {
        const progressBar = document.getElementById('tokenProgressBar');
        const progressText = document.getElementById('tokenProgressText');
        
        if (progressBar && progressText) {
            let percentage = 0;
            let text = '';
            
            switch(stage) {
                case 'generating':
                    percentage = 30;
                    text = 'Generating Q&A...';
                    break;
                case 'evaluating':
                    percentage = 70;
                    text = 'Evaluating answers...';
                    break;
                case 'complete':
                    percentage = 100;
                    text = 'Complete!';
                    break;
                default:
                    percentage = 10;
                    text = 'Starting...';
            }
            
            progressBar.style.width = `${percentage}%`;
            progressText.textContent = text;
            tokenUsageData.progressPercentage = percentage;
        }
    }

    function calculateEstimatedCost(totalTokens) {
        const costPerToken = 0.00002; // Groq pricing
        return totalTokens * costPerToken;
    }

    function estimateTokenCount(text) {
        if (!text) return 0;
        return Math.ceil(text.length / 4);
    }

    function resetTokenUsage() {
        tokenUsageData = {
            inputTokens: 0,
            outputTokens: 0,
            totalTokens: 0,
            estimatedCost: 0,
            currentStage: 'idle',
            isVisible: false,
            progressPercentage: 0
        };
        
        const container = document.getElementById('tokenUsageContainer');
        if (container) {
            container.classList.add('hidden');
            container.classList.remove('minimized', 'generating', 'evaluating', 'complete');
            container.style.transform = 'translateX(100%)';
            container.style.opacity = '0';
        }
    }


    // Initialize Application
    init();

    function init() {
        console.log('🚀 Initializing PDF Q&A Generator...');
        setupEventListeners();
        updateQuestionCountDisplay();
        showUploadSection();
        setupChunkToggle();
        setupTokenUsageCloseButton();
    }

    // ==================== EVENT LISTENERS ====================
    function setupEventListeners() {
        console.log('📝 Setting up event listeners...');

        // File Upload Events
        if (uploadArea) {
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', handleDragOver);
            uploadArea.addEventListener('dragleave', handleDragLeave);
            uploadArea.addEventListener('drop', handleDrop);
        }

        if (fileInput) {
            fileInput.addEventListener('change', handleFileSelect);
        }

        if (removeFileBtn) {
            removeFileBtn.addEventListener('click', removeFile);
        }

        // Form Events
        if (qaForm) {
            qaForm.addEventListener('submit', handleFormSubmit);
        }

        if (questionCountSlider) {
            questionCountSlider.addEventListener('input', updateQuestionCountDisplay);
        }

        // Navigation Events
        if (mcqNextBtn) {
            mcqNextBtn.addEventListener('click', handleMCQNext);
        }

        if (onelinePrevBtn) {
            onelinePrevBtn.addEventListener('click', () => goToPage('mcq'));
        }

        if (submitBtn) {
            submitBtn.addEventListener('click', handleSubmitAndEvaluate);
        }

        if (backToUploadBtn) {
            backToUploadBtn.addEventListener('click', backToUpload);
        }

        if (exportResultsBtn) {
            exportResultsBtn.addEventListener('click', exportResults);
        }

        if (retakeQuizBtn) {
            retakeQuizBtn.addEventListener('click', retakeQuiz);
        }

        console.log('✅ Event listeners set up successfully');
    }

    // ==================== FILE UPLOAD HANDLERS ====================
    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        
        if (files.length > 0) {
            if (files[0].type === 'application/pdf') {
                fileInput.files = files;
                updateFileInfo(files[0]);
            } else {
                showNotification('Please drop a valid PDF file', 'error');
            }
        }
    }

    function handleFileSelect(e) {
        if (e.target.files && e.target.files[0]) {
            updateFileInfo(e.target.files[0]);
        }
    }

    function updateFileInfo(file) {
        const fileSizeInMB = (file.size / 1024 / 1024).toFixed(2);
        
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = `${fileSizeInMB} MB`;
        
        if (fileInfo) fileInfo.classList.add('show');
        if (removeFileBtn) removeFileBtn.classList.remove('hidden');

        console.log(`📄 File selected: ${file.name} (${fileSizeInMB} MB)`);
    }

    function removeFile() {
        if (fileInput) fileInput.value = '';
        if (fileName) fileName.textContent = 'No file selected';
        if (fileSize) fileSize.textContent = '';
        if (fileInfo) fileInfo.classList.remove('show');
        if (removeFileBtn) removeFileBtn.classList.add('hidden');

        console.log('🗑️ File removed');
    }

    // ==================== QUESTION COUNT MANAGEMENT ====================
    function updateQuestionCountDisplay() {
        if (!questionCountSlider) return;

        const count = questionCountSlider.value;
        
        if (questionCountValue) questionCountValue.textContent = count;
        if (mcqCount) mcqCount.textContent = count;
        if (shortCount) shortCount.textContent = count;
    }

    // ==================== FORM SUBMISSION ====================
    async function handleFormSubmit(e) {
        e.preventDefault();
        console.log('📤 Starting form submission...');

        // Validation
        if (!fileInput || !fileInput.files[0]) {
            showNotification('Please select a PDF file', 'error');
            return;
        }

        try {
            // Show processing progress
            showProcessingProgress();
            hideUploadSection();

            // Prepare form data
            const formData = new FormData();
            formData.append('pdf_file', fileInput.files[0]);
            formData.append('profession', professionSelect ? professionSelect.value : 'general');
            formData.append('questionCount', questionCountSlider ? questionCountSlider.value : '10');
            const customPrompt = document.getElementById('customPrompt');
            formData.append('customPrompt', customPrompt ? customPrompt.value.trim() : '');

             // Estimate initial tokens
            const fileText = await getFileText(fileInput.files[0]);
            const customPromptText = customPrompt ? customPrompt.value.trim() : '';
            const inputTokens = estimateTokenCount(fileText + customPromptText);
            
            // Show initial token usage
            updateTokenDisplay(inputTokens, 0, null, 'generating');


            console.log('📊 Submitting PDF for processing...');

            // Submit to server
            const response = await fetch('/generate_qa', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            console.log('✅ Server response received:', result);


            // Update token usage with real data if available
            if (result.token_usage) {
                updateTokenDisplay(
                    result.token_usage.input_tokens || inputTokens,
                    result.token_usage.output_tokens || 0,
                    result.token_usage.estimated_cost,
                    'generating'
                );
            } else {
                // Estimate output tokens from generated content
                const outputText = (result.mcq || '') + (result.one_line || '');
                const outputTokens = estimateTokenCount(outputText);
                updateTokenDisplay(inputTokens, outputTokens, null, 'generating');
            }

            // Store result data for later use
            sessionData = {
                mcq: result.mcq || result.mcqs || result.generated_mcqs || "",
                one_line: result.one_line || result.one_lines || result.generated_one_lines || "",
                session_id: result.session_id
            };

            // Start progress polling if session ID provided
            if (result.session_id) {
                console.log(`🔄 Starting progress polling for session: ${result.session_id}`);
                pollProgressWithCompletion(result.session_id);
            } else {
                console.log('⚠️ No session ID provided, using fallback completion');
                handleImmediateCompletion();
            }

        } catch (error) {
            console.error('❌ Form submission error:', error);
            showProcessingError(error.message);
        }
    }

    // ==================== PROGRESS TRACKING ====================
    function pollProgressWithCompletion(sessionId) {
        let processingComplete = false;
        
        // Clear any existing interval
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }

        progressInterval = setInterval(async () => {
            try {
                const response = await fetch(`/progress/${sessionId}`);
                const data = await response.json();
                
                // Update progress UI
                updateProgressUI(data.progress || 0, data.status || 'Processing...', data.details || {});
                
                // Check if processing is complete
                const isComplete = (
                    data.progress >= 100 || 
                    data.status === 'Complete' || 
                    data.status.toLowerCase().includes('complete') || 
                    data.status.toLowerCase().includes('processing complete')
                );

                if (isComplete && !processingComplete) {
                    processingComplete = true;
                    console.log('✅ Processing completed successfully');
                    
                    // Show completion section
                    showProcessingCompletion(data.details || {});
                    
                    // Stop polling
                    clearInterval(progressInterval);
                    progressInterval = null;
                }
                
                // Handle errors
                if (data.status && data.status.toLowerCase().includes('error')) {
                    console.error('❌ Processing error:', data.status);
                    clearInterval(progressInterval);
                    progressInterval = null;
                    showProcessingError(data.status);
                }
                
            } catch (error) {
                console.error('❌ Progress polling error:', error);
                // Continue polling - don't stop on network errors
            }
        }, 500); // Poll every 500ms for smooth updates
    }

    function updateProgressUI(progress, status, details = {}) {
        // Ensure progress is within bounds
        progress = Math.max(0, Math.min(100, progress || 0));

        // Update main progress bar
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }
        
        if (progressPercentage) {
            progressPercentage.textContent = `${progress}%`;
        }
        
        if (progressStatus) {
            progressStatus.textContent = status || 'Processing...';
        
        }
        if (details.chunk_token_counts && details.chunk_token_counts.length > 0) {
        updateChunkTokenDetails(details.chunk_token_counts);
        }
        
        // Update detailed statistics with proper field mapping
        updateProgressStat(pageCount, details.total_pages || details.pages_processed);
        updateProgressStat(chunkCount, details.chunk_count);
        updateProgressStat(tokenUsage, details.token_estimate || details.token_usage, true); // Format with commas
        
        // Handle total questions - try multiple field combinations
        let totalQuestions = details.total_questions;
        if (totalQuestions === undefined && details.mcq_generated !== undefined && details.short_generated !== undefined) {
            totalQuestions = details.mcq_generated + details.short_generated;
        }
        updateProgressStat(questionsGenerated, totalQuestions);

        console.log(`📊 Progress: ${progress}% - ${status}`, details);
    }

    function updateProgressStat(element, value, formatNumber = false) {
        if (!element) return;

        let displayValue = '-';
        let hasValue = false;

        if (value !== undefined && value !== null && value !== '') {
            hasValue = true;
            if (formatNumber && typeof value === 'number') {
                displayValue = value.toLocaleString();
            } else {
                displayValue = String(value);
            }
        }
        
        element.textContent = displayValue;
        
        // Show/hide the stat card based on whether we have data
        const statCard = element.closest('.stat-card');
        if (statCard) {
            statCard.style.opacity = hasValue ? '1' : '0.5';
            if (hasValue) {
                // Add a subtle animation when value appears
                statCard.style.transform = 'scale(1.02)';
                setTimeout(() => {
                    statCard.style.transform = 'scale(1)';
                }, 200);
            }
        }
    }


    

    function showProcessingCompletion(details) {
        console.log('🎉 Showing processing completion');
        
        // Update final progress
        updateProgressUI(100, '✅ PDF Processing Complete!', details);

        // Create completion section HTML
        const completionHtml = `
            <div class="processing-complete">
                <div class="completion-header">
                    <i class="fas fa-check-circle"></i>
                    <h3>Analysis Complete!</h3>
                    <p>Your PDF has been successfully processed. Review the statistics above and click below to start your quiz.</p>
                </div>
                
                <div class="completion-actions">
                    <button onclick="proceedToQuestions()" class="continue-btn primary">
                        <i class="fas fa-play"></i>
                        Start Quiz
                    </button>
                    <button onclick="resetAndUploadNew()" class="continue-btn secondary">
                        <i class="fas fa-upload"></i>
                        Upload New PDF
                    </button>
                </div>
                
                <div class="completion-note">
                    <i class="fas fa-info-circle"></i>
                    <span>Take your time to review the processing statistics above</span>
                </div>
            </div>
        `;

        if (completionSection) {
            completionSection.innerHTML = completionHtml;
            completionSection.classList.remove('hidden');
        }
    }

    function handleImmediateCompletion() {
        console.log('⚡ Handling immediate completion (fallback)');
        
        setTimeout(() => {
            showProcessingCompletion({
                total_pages: '?',
                chunk_count: '?', 
                token_estimate: '?',
                total_questions: '?'
            });
        }, 1500);
    }

    // ==================== GLOBAL FUNCTIONS FOR BUTTONS ====================
    window.proceedToQuestions = function() {
        console.log('🎯 Proceeding to questions...');
        
        try {
            // Validate stored data
            if (!sessionData.mcq && !sessionData.one_line) {
                showNotification('No question data available. Please try uploading again.', 'error');
                return;
            }

            // Parse the questions
            MCQ_ITEMS = parseMCQText(sessionData.mcq);
            ONE_LINE_ITEMS = parseOneLineText(sessionData.one_line);

            console.log(`📝 Parsed ${MCQ_ITEMS.length} MCQs and ${ONE_LINE_ITEMS.length} short questions`);

            if (MCQ_ITEMS.length === 0 && ONE_LINE_ITEMS.length === 0) {
                showNotification("No questions were generated from the PDF", 'warning');
                return;
            }

            // Reset quiz state
            currentAnswers = { mcq: {}, oneline: {} };
            evaluationResults = null;

            // Render questions
            renderMCQQuestions();
            renderOnelineQuestions();
            
            // Update statistics
            updateQuizStats();

            // Show quiz section
            showQuizSection();

            showNotification(`Quiz ready! ${MCQ_ITEMS.length} MCQs and ${ONE_LINE_ITEMS.length} short questions loaded.`, 'success');
            
        } catch (error) {
            console.error('❌ Error proceeding to questions:', error);
            showNotification('Error loading questions. Please try again.', 'error');
        }
    };

    window.resetAndUploadNew = function() {
        console.log('🔄 Resetting for new upload...');
        resetQuiz();
        showUploadSection();
        showNotification('Ready for a new PDF upload!', 'info');
    };

    // ==================== QUESTION PARSING ====================
    function parseMCQText(mcqText) {
        if (!mcqText || mcqText.trim() === "") {
            console.log('⚠️ No MCQ text to parse');
            return [];
        }

        const questions = [];
        
        // Clean the text - remove headers and artifacts
        let cleanText = mcqText
            .replace(/MCQs?:\s*\n?/gi, '')
            .replace(/Multiple.{0,20}Choice.{0,20}Questions?:?\s*\n?/gi, '')
            .trim();
        
        // Split by question patterns
        const questionBlocks = cleanText.split(/(?=\n\s*\d+[\.\)])/);
        
        questionBlocks.forEach((block, index) => {
            const lines = block.trim().split('\n').map(line => line.trim()).filter(line => line);
            if (lines.length < 5) return; // Need at least question + 4 options

            // Extract question text
            let question = lines[0]
                .replace(/^\d+[\.\)]\s*/, '')
                .replace(/^Question\s*\d*[\:\.]?\s*/i, '')
                .trim();
            
            // Skip invalid questions
            if (question.length < 10 || 
                question.toLowerCase().includes('mcq') || 
                question.toLowerCase().includes('multiple choice')) {
                return;
            }

            // Extract options A), B), C), D)
            const options = {};
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i];
                const match = line.match(/^([A-D])\)\s*(.+)$/);
                if (match && match[2].trim().length > 0) {
                    options[match[1]] = match[2].trim();
                }
            }

            // Only include if we have a complete question
            if (question && Object.keys(options).length === 4) {
                questions.push({
                    id: `mcq_${questions.length + 1}`,
                    question: question,
                    options: {
                        A: options.A || '',
                        B: options.B || '',
                        C: options.C || '',
                        D: options.D || ''
                    }
                });
            }
        });

        console.log(`✅ Parsed ${questions.length} MCQ questions`);
        return questions;
    }

    function parseOneLineText(oneLineText) {
        if (!oneLineText || oneLineText.trim() === "") {
            console.log('⚠️ No short answer text to parse');
            return [];
        }

        const questions = [];
        
        // Clean text - remove headers
        let cleanText = oneLineText
            .replace(/One.{0,10}line\s+Q&A:?\s*\n?/gi, '')
            .replace(/Short\s+Answer\s+Questions?:?\s*\n?/gi, '')
            .replace(/Q&A:?\s*\n?/gi, '')
            .trim();
        
        const lines = cleanText.split('\n').filter(line => line.trim());

        lines.forEach((line, index) => {
            let question = "";
            
            // Handle various formats
            if (line.includes('Q:') && line.includes('A:')) {
                question = line.split('Q:')[1].split('A:')[0].trim();
            } else if (line.match(/^\d+[\.\)]/)) {
                const content = line.replace(/^\d+[\.\)]\s*/, '').trim();
                if (content.includes('Q:') && content.includes('A:')) {
                    question = content.split('Q:')[1].split('A:')[0].trim();
                } else if (content.length > 10 && !content.includes('A:')) {
                    question = content;
                }
            }

            // Validate and add question
            if (question && question.length > 5 && !question.toLowerCase().includes('question')) {
                questions.push({
                    id: `oneline_${questions.length + 1}`,
                    question: question
                });
            }
        });

        console.log(`✅ Parsed ${questions.length} short answer questions`);
        return questions;
    }

    // ==================== QUESTION RENDERING ====================
    function renderMCQQuestions() {
        const container = document.getElementById('mcqContainer');
        if (!container) return;
        
        if (!MCQ_ITEMS || MCQ_ITEMS.length === 0) {
            container.innerHTML = '<p class="text-center text-muted">No multiple choice questions generated.</p>';
            return;
        }

        container.innerHTML = MCQ_ITEMS.map((item, index) => `
            <div class="question-card" data-question-id="${item.id}">
                <div class="question-header">
                    <span class="question-number">Question ${index + 1} of ${MCQ_ITEMS.length}</span>
                    <span class="question-type">Multiple Choice</span>
                </div>
                <div class="question-text">${item.question}</div>
                <div class="mcq-options">
                    ${Object.entries(item.options).map(([key, value]) => `
                        <label class="mcq-option">
                            <input type="radio" name="${item.id}" value="${key}">
                            <span><strong>${key})</strong> ${value}</span>
                        </label>
                    `).join('')}
                </div>
                <div class="question-feedback" id="feedback_${item.id}"></div>
            </div>
        `).join('');

        addMCQEventListeners();
        console.log(`✅ Rendered ${MCQ_ITEMS.length} MCQ questions`);
    }

    function renderOnelineQuestions() {
        const container = document.getElementById('onelineContainer');
        if (!container) return;
        
        if (!ONE_LINE_ITEMS || ONE_LINE_ITEMS.length === 0) {
            container.innerHTML = '<p class="text-center text-muted">No short answer questions generated.</p>';
            return;
        }

        container.innerHTML = ONE_LINE_ITEMS.map((item, index) => `
            <div class="question-card" data-question-id="${item.id}">
                <div class="question-header">
                    <span class="question-number">Question ${index + 1} of ${ONE_LINE_ITEMS.length}</span>
                    <span class="question-type">Short Answer</span>
                </div>
                <div class="question-text">${item.question}</div>
                <textarea 
                    class="short-answer-input" 
                    id="answer_${item.id}" 
                    placeholder="Enter your answer here..."
                    rows="3"
                ></textarea>
                <div class="question-feedback" id="feedback_${item.id}"></div>
            </div>
        `).join('');

        addOnelineEventListeners();
        console.log(`✅ Rendered ${ONE_LINE_ITEMS.length} short answer questions`);
    }

    // ==================== EVENT LISTENERS FOR QUESTIONS ====================
    function addMCQEventListeners() {
        const radioInputs = document.querySelectorAll('#mcqContainer input[type="radio"]');
        
        radioInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                const questionId = e.target.name;
                const selectedValue = e.target.value;
                
                // Store answer
                currentAnswers.mcq[questionId] = selectedValue;
                
                // Update UI
                const questionCard = e.target.closest('.question-card');
                questionCard.classList.add('answered');
                
                // Clear previous selections
                questionCard.querySelectorAll('.mcq-option').forEach(option => {
                    option.classList.remove('selected');
                });
                
                // Mark current selection
                e.target.closest('.mcq-option').classList.add('selected');
                
                // Update statistics
                updateMCQStats();
                
                console.log(`📝 MCQ answered: ${questionId} = ${selectedValue}`);
            });
        });
    }

    function addOnelineEventListeners() {
        const textAreas = document.querySelectorAll('#onelineContainer .short-answer-input');
        
        textAreas.forEach(textarea => {
            textarea.addEventListener('input', (e) => {
                const questionId = e.target.id.replace('answer_', '');
                const answer = e.target.value.trim();
                
                // Store or remove answer
                if (answer) {
                    currentAnswers.oneline[questionId] = answer;
                    e.target.closest('.question-card').classList.add('answered');
                } else {
                    delete currentAnswers.oneline[questionId];
                    e.target.closest('.question-card').classList.remove('answered');
                }
                
                // Update statistics
                updateOnelineStats();
                
                console.log(`📝 Short answer updated: ${questionId}`);
            });
        });
    }

    // ==================== QUIZ NAVIGATION ====================
    function handleMCQNext() {
        const answeredCount = Object.keys(currentAnswers.mcq).length;
        const totalQuestions = MCQ_ITEMS.length;
        
        if (answeredCount < totalQuestions) {
            showNotification(
                `Please answer all ${totalQuestions} multiple choice questions before proceeding. You have answered ${answeredCount} out of ${totalQuestions}.`, 
                'warning'
            );
            
            // Scroll to first unanswered question
            const unansweredCard = document.querySelector('.question-card:not(.answered)');
            if (unansweredCard) {
                unansweredCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                unansweredCard.style.border = '2px solid #f59e0b';
                setTimeout(() => {
                    unansweredCard.style.border = '';
                }, 3000);
            }
            return;
        }
        
        goToPage('oneline');
    }

    function goToPage(page) {
        currentPage = page;
        
        if (page === 'mcq') {
            // Show MCQ page
            if (mcqPage) mcqPage.classList.remove('hidden');
            if (onelinePage) onelinePage.classList.add('hidden');
            
            // Update steps
            if (step1) {
                step1.classList.add('active');
                step1.classList.remove('completed');
            }
            if (step2) {
                step2.classList.remove('active');
            }
            
            updateMCQStats();
            
        } else if (page === 'oneline') {
            // Show short answer page
            if (mcqPage) mcqPage.classList.add('hidden');
            if (onelinePage) onelinePage.classList.remove('hidden');
            
            // Update steps
            if (step1) {
                step1.classList.remove('active');
                step1.classList.add('completed');
            }
            if (step2) {
                step2.classList.add('active');
            }
            
            updateOnelineStats();
        }
        
        console.log(`📄 Navigated to page: ${page}`);
    }

    function backToUpload() {
        if (confirm('Are you sure you want to go back? All progress will be lost.')) {
            resetQuiz();
            showUploadSection();
        }
    }

    function retakeQuiz() {
        if (confirm('Are you sure you want to retake the quiz? All answers will be reset.')) {
            // Reset answers but keep questions
            currentAnswers = { mcq: {}, oneline: {} };
            evaluationResults = null;
            
            // Clear UI feedback
            document.querySelectorAll('.question-feedback').forEach(feedback => {
                feedback.style.display = 'none';
                feedback.innerHTML = '';
            });
            
            // Reset question states
            document.querySelectorAll('.question-card').forEach(card => {
                card.classList.remove('answered');
            });
            
            // Clear form inputs
            document.querySelectorAll('input[type="radio"]').forEach(radio => {
                radio.checked = false;
            });
            
            document.querySelectorAll('.short-answer-input').forEach(textarea => {
                textarea.value = '';
            });
            
            document.querySelectorAll('.mcq-option').forEach(option => {
                option.classList.remove('selected');
            });
            
            // Hide results
            if (finalSummary) finalSummary.classList.add('hidden');
            if (actionButtons) actionButtons.classList.add('hidden');
            
            // Go back to first page
            goToPage('mcq');
            
            showNotification('Quiz reset! You can start answering again.', 'info');
        }
    }

    // ==================== STATISTICS UPDATES ====================
    function updateQuizStats() {
        updateMCQStats();
        updateOnelineStats();
    }

    function updateMCQStats() {
        const answered = Object.keys(currentAnswers.mcq).length;
        const total = MCQ_ITEMS.length;
        
        if (mcqAnswered) mcqAnswered.textContent = answered;
        if (mcqTotal) mcqTotal.textContent = total;
        
        // Update next button
        if (mcqNextBtn) {
            if (answered === total && total > 0) {
                mcqNextBtn.innerHTML = `
                    <span><i class="fas fa-check"></i> All Complete - Continue</span>
                    <i class="fas fa-arrow-right"></i>
                `;
                mcqNextBtn.disabled = false;
                mcqNextBtn.classList.remove('btn-disabled');
            } else {
                mcqNextBtn.innerHTML = `
                    <span>Complete All Questions (${answered}/${total})</span>
                    <i class="fas fa-lock"></i>
                `;
                mcqNextBtn.disabled = true;
                mcqNextBtn.classList.add('btn-disabled');
            }
        }
    }
    
    function updateOnelineStats() {
        const answered = Object.keys(currentAnswers.oneline).length;
        const total = ONE_LINE_ITEMS.length;
        
        if (onelineAnswered) onelineAnswered.textContent = answered;
        if (onelineTotal) onelineTotal.textContent = total;
        
        // Update submit button
        if (submitBtn) {
            const btnSpan = submitBtn.querySelector('span');
            if (btnSpan) {
                if (answered === total && total > 0) {
                    btnSpan.innerHTML = '<i class="fas fa-check"></i> All Complete - Submit & Evaluate';
                    submitBtn.disabled = false;
                    submitBtn.classList.remove('btn-disabled');
                } else {
                    btnSpan.textContent = `Complete All Questions (${answered}/${total}) to Submit`;
                    submitBtn.disabled = true;
                    submitBtn.classList.add('btn-disabled');
                }
            }
        }
    }

    // ==================== SUBMISSION AND EVALUATION ====================
    async function handleSubmitAndEvaluate() {
        console.log('📤 Starting evaluation...');
        
        const mcqAnswers = collectMCQAnswers();
        const oneLineAnswers = collectOneLineAnswers();
        
        // Validation
        if (oneLineAnswers.length < ONE_LINE_ITEMS.length) {
            const missing = ONE_LINE_ITEMS.length - oneLineAnswers.length;
            showNotification(
                `Please answer all ${ONE_LINE_ITEMS.length} short answer questions before submitting. ${missing} question(s) remaining.`, 
                'warning'
            );
            
            // Scroll to first unanswered question
            const unansweredCard = document.querySelector('#onelineContainer .question-card:not(.answered)');
            if (unansweredCard) {
                unansweredCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                unansweredCard.style.border = '2px solid #f59e0b';
                setTimeout(() => {
                    unansweredCard.style.border = '';
                }, 3000);
            }
            return;
        }

        if (mcqAnswers.length === 0 && oneLineAnswers.length === 0) {
            showNotification('No answers to evaluate', 'error');
            return;
        }

        try {
            showSubmitLoading(true);

            const response = await fetch('/evaluate_answers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mcqs: mcqAnswers,
                    one_lines: oneLineAnswers
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            evaluationResults = await response.json();

            // Show results
            showInlineFeedback(evaluationResults);
            showFinalSummary(evaluationResults);

            // Show action buttons
            if (actionButtons) {
                actionButtons.classList.remove('hidden');
            }

            showSubmitLoading(false);
            showNotification('Answers evaluated successfully! Check feedback below each question.', 'success');

        } catch (error) {
            showSubmitLoading(false);
            showNotification('Error evaluating answers. Please try again.', 'error');
            console.error('❌ Evaluation error:', error);
        }
    }

    function collectMCQAnswers() {
        return Object.entries(currentAnswers.mcq).map(([questionId, selectedOption]) => {
            const question = MCQ_ITEMS.find(q => q.id === questionId);
            return {
                id: questionId,
                question: question?.question || '',
                options: question?.options || {},
                selected: [selectedOption]
            };
        });
    }

    function collectOneLineAnswers() {
        return Object.entries(currentAnswers.oneline).map(([questionId, answer]) => {
            const question = ONE_LINE_ITEMS.find(q => q.id === questionId);
            return {
                id: questionId,
                question: question?.question || '',
                answer: answer
            };
        });
    }

    // ==================== RESULTS DISPLAY ====================
    function showInlineFeedback(results) {
        // Handle MCQ feedback
        if (results.mcqs) {
            Object.entries(results.mcqs).forEach(([qid, result]) => {
                const feedbackDiv = document.getElementById(`feedback_${qid}`);
                if (feedbackDiv) {
                    const statusClass = result.status || 'partial';
                    feedbackDiv.className = `question-feedback ${statusClass}`;
                    feedbackDiv.style.display = 'block';

                    const statusIcons = {
                        correct: '✅',
                        incorrect: '❌',
                        partial: '⚠️'
                    };

                    const statusTexts = {
                        correct: 'Correct',
                        incorrect: 'Incorrect',
                        partial: 'Partial Credit'
                    };

                    let correctAnswerHtml = '';
                    if (result.correct_options && result.correct_options.length > 0) {
                        correctAnswerHtml = `<div class="correct-answer">
                            <strong>Correct Answer(s):</strong> ${result.correct_options.join(', ')}
                        </div>`;
                    }

                    feedbackDiv.innerHTML = `
                        <div class="feedback-status">
                            <span>${statusIcons[statusClass]} ${statusTexts[statusClass]}</span>
                        </div>
                        <div class="feedback-explanation">${result.explanation || ''}</div>
                        ${correctAnswerHtml}
                    `;
                }
            });
        }

        // Handle One-line feedback
        if (results.one_lines) {
            Object.entries(results.one_lines).forEach(([qid, result]) => {
                const feedbackDiv = document.getElementById(`feedback_${qid}`);
                if (feedbackDiv) {
                    const statusClass = result.status || 'partial';
                    feedbackDiv.className = `question-feedback ${statusClass}`;
                    feedbackDiv.style.display = 'block';

                    const statusIcons = {
                        correct: '✅',
                        incorrect: '❌',
                        partial: '⚠️'
                    };

                    const statusTexts = {
                        correct: 'Correct',
                        incorrect: 'Incorrect',
                        partial: 'Partial Credit'
                    };

                    let modelAnswerHtml = '';
                    if (result.model_answer) {
                        modelAnswerHtml = `<div class="correct-answer">
                            <strong>Model Answer:</strong> ${result.model_answer}
                        </div>`;
                    }

                    feedbackDiv.innerHTML = `
                        <div class="feedback-status">
                            <span>${statusIcons[statusClass]} ${statusTexts[statusClass]}</span>
                        </div>
                        <div class="feedback-explanation">${result.explanation || ''}</div>
                        ${modelAnswerHtml}
                    `;
                }
            });
        }
    }

    function showFinalSummary(results) {
        if (!finalSummary) return;
        
        const mcqResults = Object.values(results.mcqs || {});
        const oneLineResults = Object.values(results.one_lines || {});

        const totalQuestions = mcqResults.length + oneLineResults.length;
        const correctAnswers = [...mcqResults, ...oneLineResults].filter(r => r.status === 'correct').length;
        const partialAnswers = [...mcqResults, ...oneLineResults].filter(r => r.status === 'partial').length;
        const incorrectAnswers = totalQuestions - correctAnswers - partialAnswers;

        const percentage = totalQuestions > 0 ? Math.round((correctAnswers / totalQuestions) * 100) : 0;

        let performanceMessage = '';
        let performanceIcon = '';
        if (percentage >= 80) {
            performanceMessage = 'Excellent work! Keep it up!';
            performanceIcon = '🎉';
        } else if (percentage >= 60) {
            performanceMessage = 'Good job! You\'re on the right track!';
            performanceIcon = '👍';
        } else {
            performanceMessage = 'Keep studying! You can improve!';
            performanceIcon = '📚';
        }

        finalSummary.classList.remove('hidden');
        finalSummary.innerHTML = `
            <h3><i class="fas fa-trophy"></i> Final Results Summary</h3>
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">${totalQuestions}</div>
                    <div class="stat-label">Total Questions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: var(--success)">${correctAnswers}</div>
                    <div class="stat-label">Correct</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: var(--warning)">${partialAnswers}</div>
                    <div class="stat-label">Partial</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: var(--error)">${incorrectAnswers}</div>
                    <div class="stat-label">Incorrect</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: var(--primary-color)">${percentage}%</div>
                    <div class="stat-label">Overall Score</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 1.5rem; padding: 1rem; background: var(--secondary-color); border-radius: var(--radius-md);">
                <p style="font-size: 1.2rem; color: var(--text-primary); font-weight: 500;">
                    ${performanceIcon} ${performanceMessage}
                </p>
            </div>
        `;
    }

    // ==================== UTILITY FUNCTIONS ====================
    function showUploadSection() {
        if (uploadSection) uploadSection.classList.remove('hidden');
        if (processingProgress) processingProgress.classList.add('hidden');
        if (quizSection) quizSection.classList.add('hidden');
        currentPage = 'upload';
        console.log('📤 Showing upload section');
    }

    function hideUploadSection() {
        if (uploadSection) uploadSection.classList.add('hidden');
    }

    function showProcessingProgress() {
        if (processingProgress) {
            processingProgress.classList.remove('hidden');
            resetProgressSection();
        }
        console.log('⏳ Showing processing progress');
    }

    function showQuizSection() {
        if (uploadSection) uploadSection.classList.add('hidden');
        if (processingProgress) processingProgress.classList.add('hidden');
        if (quizSection) quizSection.classList.remove('hidden');
        
        // Show MCQ page by default
        goToPage('mcq');
        console.log('🎯 Showing quiz section');
    }

    function resetProgressSection() {
        // Reset completion section
        if (completionSection) {
            completionSection.innerHTML = '';
            completionSection.classList.add('hidden');
        }
        
        // Reset progress values
        updateProgressUI(0, 'Starting...', {});
        
        // Reset stat cards opacity
        document.querySelectorAll('.stat-card').forEach(card => {
            card.style.opacity = '0.5';
        });
    }

    function showProcessingError(errorMessage) {
        console.error('❌ Processing error:', errorMessage);
        
        // Hide progress, show upload
        showUploadSection();
        
        // Show error notification
        showNotification(`Processing failed: ${errorMessage}`, 'error');
    }

    function showSubmitLoading(show) {
        if (submitBtn) {
            submitBtn.disabled = show;
            const btnText = submitBtn.querySelector('span');
            
            if (btnText) {
                if (show) {
                    btnText.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Evaluating...';
                } else {
                    btnText.innerHTML = '<i class="fas fa-check"></i> All Complete - Submit & Evaluate';
                }
            }
        }
    }

    function exportResults() {
        if (!evaluationResults) {
            showNotification('No results to export', 'warning');
            return;
        }

        const exportData = {
            summary: {
                totalQuestions: MCQ_ITEMS.length + ONE_LINE_ITEMS.length,
                mcqQuestions: MCQ_ITEMS.length,
                onelineQuestions: ONE_LINE_ITEMS.length,
                mcqAnswered: Object.keys(currentAnswers.mcq).length,
                onelineAnswered: Object.keys(currentAnswers.oneline).length,
                timestamp: new Date().toISOString(),
                profession: professionSelect ? professionSelect.value : 'general',
                questionCount: questionCountSlider ? questionCountSlider.value : '10'
            },
            questions: {
                mcq: MCQ_ITEMS,
                oneline: ONE_LINE_ITEMS
            },
            answers: currentAnswers,
            results: evaluationResults
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `quiz_results_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showNotification('Results exported successfully!', 'success');
    }

    function resetQuiz() {
        console.log('🔄 Resetting quiz...');
        
        // Clear data
        MCQ_ITEMS = [];
        ONE_LINE_ITEMS = [];
        currentAnswers = { mcq: {}, oneline: {} };
        evaluationResults = null;
        sessionData = {};
        
        // Clear file
        removeFile();
        
        // Clear progress polling
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
        
        // Clear question containers
        const mcqContainer = document.getElementById('mcqContainer');
        const onelineContainer = document.getElementById('onelineContainer');
        if (mcqContainer) mcqContainer.innerHTML = '';
        if (onelineContainer) onelineContainer.innerHTML = '';
        
        // Hide results
        if (finalSummary) finalSummary.classList.add('hidden');
        if (actionButtons) actionButtons.classList.add('hidden');
        
        // Reset stats
        if (mcqAnswered) mcqAnswered.textContent = '0';
        if (mcqTotal) mcqTotal.textContent = '0';
        if (onelineAnswered) onelineAnswered.textContent = '0';
        if (onelineTotal) onelineTotal.textContent = '0';
        
        currentPage = 'upload';
    }

    function showNotification(message, type = 'info') {
        console.log(`📢 ${type.toUpperCase()}: ${message}`);
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">
                    ${type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️'}
                </span>
                <span class="notification-text">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        const container = document.getElementById('notifications');
        if (container) {
            container.appendChild(notification);

            // Auto remove after 7 seconds
            setTimeout(() => {
                if (container.contains(notification)) {
                    notification.style.animation = 'slideOutRight 0.3s ease-out';
                    setTimeout(() => {
                        if (container.contains(notification)) {
                            container.removeChild(notification);
                        }
                    }, 300);
                }
            }, 7000);
        }
    }


    // Add this function to handle chunk token details display
function updateChunkTokenDetails(chunkTokenCounts) {
    const chunkDetails = document.getElementById('chunkDetails');
    const chunkGrid = document.getElementById('chunkGrid');
    const totalChunks = document.getElementById('totalChunks');
    const avgTokens = document.getElementById('avgTokens');
    const tokenRange = document.getElementById('tokenRange');
    
    if (!chunkTokenCounts || chunkTokenCounts.length === 0) {
        return;
    }
    
    // Show chunk details section
    if (chunkDetails) {
        chunkDetails.classList.remove('hidden');
    }
    
    // Calculate statistics
    const tokenCounts = chunkTokenCounts.map(chunk => chunk.token_count);
    const totalTokens = tokenCounts.reduce((sum, tokens) => sum + tokens, 0);
    const avgTokenCount = Math.round(totalTokens / tokenCounts.length);
    const minTokens = Math.min(...tokenCounts);
    const maxTokens = Math.max(...tokenCounts);
    
    // Update summary
    if (totalChunks) totalChunks.textContent = chunkTokenCounts.length;
    if (avgTokens) avgTokens.textContent = avgTokenCount.toLocaleString();
    if (tokenRange) tokenRange.textContent = `${minTokens}-${maxTokens}`;
    
    // Create chunk grid
    const chunkItems = chunkTokenCounts.map(chunk => {
        const percentage = ((chunk.token_count / maxTokens) * 100);
        const intensity = Math.min(percentage / 100, 1); // For color intensity
        
        return `
            <div class="chunk-item" data-chunk="${chunk.chunk_number}">
                <div class="chunk-header">
                    <span class="chunk-num">${chunk.chunk_number}</span>
                    <span class="chunk-tokens">${chunk.token_count.toLocaleString()}</span>
                </div>
                <div class="chunk-bar">
                    <div class="chunk-fill" style="width: ${percentage}%; opacity: ${0.4 + intensity * 0.6}"></div>
                </div>
                <div class="chunk-words">${chunk.word_count} words</div>
            </div>
        `;
    }).join('');
    
    if (chunkGrid) {
        chunkGrid.innerHTML = chunkItems;
    }
    
    console.log(`📊 Chunk details updated: ${chunkTokenCounts.length} chunks`);
}

// Add toggle functionality for chunk details
function setupChunkToggle() {
    const chunkToggle = document.getElementById('chunkToggle');
    const chunkDetailsContent = document.getElementById('chunkDetailsContent');
    
    if (chunkToggle && chunkDetailsContent) {
        chunkToggle.addEventListener('click', () => {
            const isCollapsed = chunkDetailsContent.classList.contains('collapsed');
            chunkDetailsContent.classList.toggle('collapsed');
            chunkToggle.textContent = isCollapsed ? 'Hide Details' : 'Show Details';
        });
    }
}

// Helper function to get text from file for token estimation
async function getFileText(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            // Estimate text content from file size (rough approximation)
            resolve('A'.repeat(Math.min(content.length / 10, 5000)));
        };
        reader.readAsArrayBuffer(file);
    });
}




    // ==================== INITIALIZATION COMPLETE ====================
    console.log('✅ PDF Q&A Generator initialized successfully');
});