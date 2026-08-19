document.addEventListener('DOMContentLoaded', () => {
    // --- DOM ELEMENTS ---
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const messagesContainer = document.getElementById('messagesContainer');
    const typingIndicator = document.getElementById('typingIndicator');
    const sendBtn = document.querySelector('.send-btn');
    const micBtn = document.querySelector('.mic-btn'); 
    const chatArea = document.getElementById('chatArea');

    // --- STATE MANAGEMENT ---
    let isProcessing = false;
    let sessionId = localStorage.getItem('session_id') || generateUUID();
    
    // Save session ID if new
    if (!localStorage.getItem('session_id')) {
        localStorage.setItem('session_id', sessionId);
    }

    // --- SPEECH RECOGNITION SETUP ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isListening = false;
    let silenceTimer = null;
    let restartTimer = null;
    let finalTranscript = '';

    const SILENCE_TIMEOUT_MS = 1500;
    const RESTART_DELAY_MS = 250;
    const MAX_LISTEN_WITHOUT_SPEECH_MS = 30000;
    let maxSilenceTimer = null;

    async function requestMicrophoneAccess() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            // Can't pre-check permission on this browser; let SpeechRecognition try directly.
            return { granted: true };
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // We only needed this to trigger/confirm the permission prompt.
            // SpeechRecognition manages its own capture, so release this stream.
            stream.getTracks().forEach((track) => track.stop());
            return { granted: true };
        } catch (error) {
            return { granted: false, error };
        }
    }

    function describeMicrophoneError(error) {
        switch (error && error.name) {
            case 'NotAllowedError':
            case 'SecurityError':
                return 'Mic blocked. Allow it in your browser and OS settings.';
            case 'NotFoundError':
                return 'No microphone was found on this device.';
            case 'NotReadableError':
                return 'Microphone is in use by another app.';
            default:
                return 'Unable to access the microphone. Check browser/OS settings.';
        }
    }

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;

        function clearSilenceTimer() {
            window.clearTimeout(silenceTimer);
            silenceTimer = null;
        }

        function clearRestartTimer() {
            window.clearTimeout(restartTimer);
            restartTimer = null;
        }

        function clearMaxSilenceTimer() {
            window.clearTimeout(maxSilenceTimer);
            maxSilenceTimer = null;
        }

        function scheduleMaxSilenceTimeout() {
            clearMaxSilenceTimer();
            maxSilenceTimer = window.setTimeout(() => {
                // No speech at all across 30s of restart cycling — give up
                // for real instead of listening silently forever.
                isListening = false;
                clearSilenceTimer();
                clearRestartTimer();
                recognition.stop();
                userInput.placeholder = "No speech detected. Tap the mic to try again.";
            }, MAX_LISTEN_WITHOUT_SPEECH_MS);
        }

        function updateTranscript(transcript) {
            userInput.value = transcript.trim();
            userInput.style.height = 'auto';
            userInput.style.height = `${userInput.scrollHeight}px`;
            sendBtn.toggleAttribute('disabled', !userInput.value);
        }

        function stopAfterSilence() {
            clearSilenceTimer();
            silenceTimer = window.setTimeout(() => {
                isListening = false;
                recognition.stop();
            }, SILENCE_TIMEOUT_MS);
        }

        function restartRecognition() {
            if (!isListening) {
                return;
            }

            clearRestartTimer();
            restartTimer = window.setTimeout(() => {
                if (!isListening) {
                    return;
                }

                try {
                    recognition.start();
                } catch (error) {
                    console.error('Unable to restart speech recognition:', error);
                }
            }, RESTART_DELAY_MS);
        }

        recognition.onstart = () => {
            micBtn.classList.add('recording');
            micBtn.style.color = '#ef4444';
            userInput.placeholder = "Listening...";
        };

        recognition.onend = () => {
            if (isListening) {
                restartRecognition();
                return;
            }

            clearSilenceTimer();
            clearRestartTimer();
            clearMaxSilenceTimer();
            micBtn.classList.remove('recording');
            micBtn.style.color = '';
            userInput.placeholder = "Type your question here...";
        };

        recognition.onresult = (event) => {
            clearSilenceTimer();
            scheduleMaxSilenceTimeout();

            let interimTranscript = '';
            let receivedFinalResult = false;

            for (let index = event.resultIndex; index < event.results.length; index += 1) {
                const result = event.results[index];
                const transcript = result[0].transcript;

                if (result.isFinal) {
                    finalTranscript += `${transcript} `;
                    receivedFinalResult = true;
                } else {
                    interimTranscript += transcript;
                }
            }

            updateTranscript(`${finalTranscript} ${interimTranscript}`);
            userInput.focus();

            if (receivedFinalResult) {
                stopAfterSilence();
            }
        };

        recognition.onerror = (event) => {
            console.error("Speech Error:", event.error);

            if (event.error === 'no-speech') {
                // Chrome stops the recognizer after a short silence window
                // even though the mic is still open and continuous=true.
                // Leave isListening as-is so the onend handler above
                // restarts recognition automatically — this is what makes
                // tapping the mic once keep waiting for speech instead of
                // giving up after a few seconds of silence.
                return;
            }

            isListening = false;
            clearSilenceTimer();
            clearRestartTimer();
            clearMaxSilenceTimer();
            micBtn.classList.remove('recording');
            micBtn.style.color = '';

            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                userInput.placeholder = "Microphone access is required.";
            } else if (event.error === 'audio-capture') {
                userInput.placeholder = "No microphone was found.";
            } else if (event.error === 'network') {
                // Chrome's speech recognition is cloud-based: it sends audio
                // to Google's servers, so this means that connection failed
                // (no internet, VPN/firewall blocking it, DNS issue, etc.) —
                // it is not something this app's code can fix.
                userInput.placeholder = "Network error. Check your internet connection and try again.";
            } else {
                userInput.placeholder = "Unable to recognize speech. Try again.";
            }
        };

        micBtn.addEventListener('click', async () => {
            if (isListening) {
                isListening = false;
                clearSilenceTimer();
                clearMaxSilenceTimer();
                recognition.stop();
                return;
            }

            micBtn.setAttribute('disabled', 'true');
            const permission = await requestMicrophoneAccess();
            micBtn.removeAttribute('disabled');

            if (!permission.granted) {
                console.error('Microphone permission error:', permission.error);
                userInput.placeholder = describeMicrophoneError(permission.error);
                return;
            }

            finalTranscript = userInput.value.trim();
            isListening = true;

            try {
                recognition.start();
                scheduleMaxSilenceTimeout();
            } catch (error) {
                isListening = false;
                console.error('Unable to start speech recognition:', error);
                userInput.placeholder = "Unable to start microphone.";
            }
        });

    } else {
        console.log("Speech Recognition not supported.");
        micBtn.style.display = 'none'; 
    }


    // --- HELPER FUNCTIONS ---
    
    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    function scrollToBottom() {
        chatArea.scrollTo({
            top: chatArea.scrollHeight,
            behavior: 'smooth'
        });
    }

    function createMessageElement(text, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

        if (!isUser) {
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'avatar';
            // UPDATED: Uses the Shield Icon instead of the Robot
            avatarDiv.innerHTML = '<i class="fa-solid fa-user-shield"></i>';
            messageDiv.appendChild(avatarDiv);
        }

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'bubble';
        bubbleDiv.textContent = text;
        
        messageDiv.appendChild(bubbleDiv);
        return messageDiv;
    }

    // --- EVENT LISTENERS ---

    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        if (this.value.trim().length > 0) {
            sendBtn.removeAttribute('disabled');
        } else {
            sendBtn.setAttribute('disabled', 'true');
        }
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const question = userInput.value.trim();
        if (!question || isProcessing) return;

        isProcessing = true;
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.setAttribute('disabled', 'true');

        messagesContainer.appendChild(createMessageElement(question, true));
        scrollToBottom();

        typingIndicator.classList.remove('hidden');
        scrollToBottom();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId 
                },
                body: JSON.stringify({ 
                    question: question,
                    session_id: sessionId 
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            typingIndicator.classList.add('hidden');

            if (data.answer) {
                messagesContainer.appendChild(createMessageElement(data.answer, false));
            } else if (data.error) {
                messagesContainer.appendChild(createMessageElement("Error: " + data.error, false));
            } else {
                messagesContainer.appendChild(createMessageElement("I'm not sure how to answer that.", false));
            }

        } catch (error) {
            console.error("Fetch error:", error);
            typingIndicator.classList.add('hidden');
            messagesContainer.appendChild(createMessageElement("Sorry, server error.", false));
        } finally {
            isProcessing = false;
            scrollToBottom();
            userInput.focus();
        }
    });
});
