// Initialize Vanta.js background effect
document.addEventListener('DOMContentLoaded', function() {
  VANTA.HALO({
    el: "#vanta-bg",
    baseColor: 0xff00ff,
    color2: 0x00ffff,
    backgroundColor: 0x0f0f0f,
    mouseControls: true,
    touchControls: true,
    gyroControls: false,
    scale: 1.0,
    scaleMobile: 1.0
  });
  
  // Initialize chat functionality
  initializeChat();
});

// Chat functionality
function initializeChat() {
  const chatMessages = document.getElementById('chat-messages');
  const userInput = document.getElementById('user-input');
  const sendButton = document.getElementById('send-button');
  
  // Add initial bot message
  addMessage("Hello! I'm Omnis. How can I help you today?", 'bot');
  
  // Event listeners
  sendButton.addEventListener('click', sendMessage);
  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  });
  
  // Hamburger menu toggle
  const hamburger = document.getElementById('hamburger');
  const menuDialog = document.querySelector('.menu-dialog');
  
  hamburger.addEventListener('click', () => {
    const isVisible = menuDialog.style.visibility === 'visible';
    menuDialog.style.opacity = isVisible ? '0' : '1';
    menuDialog.style.visibility = isVisible ? 'hidden' : 'visible';
  });
  
  // Menu item handlers
  const menuItems = document.querySelectorAll('.menu-dialog ul li');
  menuItems.forEach(item => {
    item.addEventListener('click', () => {
      const action = item.textContent.toLowerCase();
      
      if (action === 'clear chat') {
        chatMessages.innerHTML = '';
        addMessage("Chat cleared. How can I help you?", 'bot');
      } else if (action === 'about') {
        addMessage("I'm a chatbot powered by AI. I can help you with various tasks.", 'bot');
      } else if (action === 'help') {
        addMessage("You can ask me questions or give me commands. I'll do my best to help!", 'bot');
      } else if (action === 'settings') {
        addMessage("Settings functionality coming soon!", 'bot');
      }
      
      // Hide menu after action
      menuDialog.style.opacity = '0';
      menuDialog.style.visibility = 'hidden';
    });
  });
  
  function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    userInput.value = '';
    
    // Show typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'chat-message bot typing';
    typingIndicator.textContent = 'Thinking...';
    chatMessages.appendChild(typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Send message to backend
    fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: message,
        agent: 'auto'
      })
    })
    .then(response => response.json())
    .then(data => {
      // Remove typing indicator
      chatMessages.removeChild(typingIndicator);
      
      // Add bot response
      addMessage(data.response, 'bot');
      
      // Add usage info if available
      if (data.usage) {
        const usageInfo = document.createElement('div');
        usageInfo.className = 'usage-info';
        usageInfo.textContent = `Usage: ${data.usage.tokens} tokens ($${data.usage.cost}) | Agent: ${data.agent}`;
        chatMessages.appendChild(usageInfo);
      }
      
      chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => {
      // Remove typing indicator
      chatMessages.removeChild(typingIndicator);
      
      // Add error message
      addMessage("Sorry, I encountered an error. Please try again.", 'bot');
      console.error('Error:', error);
    });
  }
  
  function addMessage(text, sender) {
    const messageElement = document.createElement('div');
    messageElement.className = `chat-message ${sender}`;
    messageElement.textContent = text;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
} 