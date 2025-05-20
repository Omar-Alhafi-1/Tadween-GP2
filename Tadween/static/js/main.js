/**
 * Initialize the application
 */
function initApp() {
    // Get DOM elements
    const questionForm = document.getElementById('question-form');
    const questionInput = document.getElementById('question');
    const submitBtn = document.getElementById('submit-btn');
    const answerContainer = document.getElementById('answer-container');
    const answerText = document.getElementById('answer-text');
    const sourcesList = document.getElementById('sources-list');
    const metricsContainer = document.getElementById('metrics-container');
    const bleuScore = document.getElementById('bleu-score');
    const bertScore = document.getElementById('bert-score');
    
    const testDataFile = document.getElementById('test-data-file');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const evaluationResults = document.getElementById('evaluation-results');
    const avgBleuScore = document.getElementById('avg-bleu-score');
    const avgBertScore = document.getElementById('avg-bert-score');
    const evaluationTableBody = document.getElementById('evaluation-table-body');

    // Add event listeners, but only if elements exist
    if (questionForm) {
        questionForm.addEventListener('submit', handleQuestionSubmit);
    }
    
    if (testDataFile) {
        testDataFile.addEventListener('change', handleFileChange);
    }
    
    if (evaluateBtn) {
        evaluateBtn.addEventListener('click', handleEvaluateClick);
    }

    /**
     * Handle question form submission
     * @param {Event} event - The form submit event
     */
    function handleQuestionSubmit(event) {
        event.preventDefault();
        
        // Check if elements exist
        if (!questionInput || !submitBtn || !answerContainer) {
            console.error('Required DOM elements not found');
            return;
        }
        
        const question = questionInput.value.trim();
        if (!question) {
            return;
        }

        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> جاري المعالجة...';
        
        // Hide previous answer
        answerContainer.classList.add('d-none');
        
        // Send request to the server
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                use_legacy: false // Use enhanced agent chain
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Check if elements exist
            if (!answerText || !sourcesList) {
                console.error('Required DOM elements not found in response handler');
                return;
            }
            
            // Show spelling correction if available
            if (data.corrected_question) {
                const correctionAlert = document.createElement('div');
                correctionAlert.className = 'alert alert-info mb-3';
                correctionAlert.innerHTML = `<strong>تم تصحيح السؤال:</strong> ${data.corrected_question}`;
                answerText.innerHTML = '';
                answerText.appendChild(correctionAlert);
                answerText.innerHTML += data.answer;
            } else {
                // Show the answer without correction
                answerText.innerHTML = data.answer;
            }
            
            // Clear and populate sources
            sourcesList.innerHTML = '';
            if (data.chunks && data.chunks.length > 0) {
                data.chunks.forEach(chunk => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'list-group-item';
                    sourceItem.innerHTML = `<strong>${chunk.metadata.article || 'مصدر'}</strong>: ${chunk.text}`;
                    sourcesList.appendChild(sourceItem);
                });
            } else {
                const noSourcesItem = document.createElement('div');
                noSourcesItem.className = 'list-group-item';
                noSourcesItem.textContent = 'لم يتم العثور على مصادر محددة';
                sourcesList.appendChild(noSourcesItem);
            }
            
            // Show metrics if available
            if (data.metrics && (data.metrics.bleu_score !== undefined || data.metrics.bert_score !== undefined)) {
                metricsContainer.classList.remove('d-none');
                bleuScore.textContent = data.metrics.bleu_score !== undefined 
                    ? data.metrics.bleu_score.toFixed(4) 
                    : '-';
                bertScore.textContent = data.metrics.bert_score !== undefined 
                    ? data.metrics.bert_score.toFixed(4) 
                    : '-';
            } else {
                metricsContainer.classList.add('d-none');
            }
            
            // Show the answer container
            answerContainer.classList.remove('d-none');
        })
        .catch(error => {
            console.error('Error:', error);
            answerText.innerHTML = `<div class="alert alert-danger">حدث خطأ أثناء معالجة السؤال: ${error.message}</div>`;
            answerContainer.classList.remove('d-none');
        })
        .finally(() => {
            // Reset button state
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'إرسال السؤال <i class="fas fa-paper-plane ms-2"></i>';
        });
    }

    /**
     * Handle file input change
     * @param {Event} event - The file input change event
     */
    function handleFileChange(event) {
        const file = event.target.files[0];
        if (file) {
            evaluateBtn.disabled = false;
        } else {
            evaluateBtn.disabled = true;
        }
    }

    /**
     * Handle evaluate button click
     */
    function handleEvaluateClick() {
        const file = testDataFile.files[0];
        if (!file) {
            return;
        }

        // Show loading state
        evaluateBtn.disabled = true;
        evaluateBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> جاري التقييم...';
        
        // Hide previous results
        evaluationResults.classList.add('d-none');
        
        // Read file content
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const testData = JSON.parse(e.target.result);
                
                // Send test data to server
                fetch('/evaluate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        test_data: testData
                    }),
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    // Display average scores
                    avgBleuScore.textContent = data.metrics.average_bleu_score.toFixed(4);
                    avgBertScore.textContent = data.metrics.average_bert_score.toFixed(4);
                    
                    // Clear and populate results table
                    evaluationTableBody.innerHTML = '';
                    if (data.results && data.results.length > 0) {
                        data.results.forEach((result, index) => {
                            const row = document.createElement('tr');
                            
                            const indexCell = document.createElement('td');
                            indexCell.textContent = index + 1;
                            row.appendChild(indexCell);
                            
                            const questionCell = document.createElement('td');
                            questionCell.textContent = result.question;
                            row.appendChild(questionCell);
                            
                            const groundTruthCell = document.createElement('td');
                            groundTruthCell.textContent = result.ground_truth;
                            row.appendChild(groundTruthCell);
                            
                            const predictionCell = document.createElement('td');
                            predictionCell.textContent = result.prediction;
                            row.appendChild(predictionCell);
                            
                            const bleuCell = document.createElement('td');
                            bleuCell.textContent = result.bleu_score.toFixed(4);
                            row.appendChild(bleuCell);
                            
                            const bertCell = document.createElement('td');
                            bertCell.textContent = result.bert_score.toFixed(4);
                            row.appendChild(bertCell);
                            
                            evaluationTableBody.appendChild(row);
                        });
                    } else {
                        const row = document.createElement('tr');
                        const cell = document.createElement('td');
                        cell.colSpan = 6;
                        cell.textContent = 'لم يتم العثور على نتائج';
                        cell.className = 'text-center';
                        row.appendChild(cell);
                        evaluationTableBody.appendChild(row);
                    }
                    
                    // Show results
                    evaluationResults.classList.remove('d-none');
                })
                .catch(error => {
                    console.error('Error:', error);
                    evaluationTableBody.innerHTML = '';
                    const row = document.createElement('tr');
                    const cell = document.createElement('td');
                    cell.colSpan = 6;
                    cell.innerHTML = `<div class="alert alert-danger">حدث خطأ أثناء التقييم: ${error.message}</div>`;
                    cell.className = 'text-center';
                    row.appendChild(cell);
                    evaluationTableBody.appendChild(row);
                    
                    evaluationResults.classList.remove('d-none');
                })
                .finally(() => {
                    // Reset button state
                    evaluateBtn.disabled = false;
                    evaluateBtn.innerHTML = 'بدء التقييم <i class="fas fa-chart-bar ms-2"></i>';
                });
            } catch (error) {
                console.error('Error parsing JSON:', error);
                alert('خطأ في تنسيق ملف JSON. يرجى التأكد من صحة التنسيق.');
                
                // Reset button state
                evaluateBtn.disabled = false;
                evaluateBtn.innerHTML = 'بدء التقييم <i class="fas fa-chart-bar ms-2"></i>';
            }
        };
        reader.readAsText(file);
    }
}
