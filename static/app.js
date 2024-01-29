function isPhotoLink(message) {
    const imageKeywords = ['jpg', 'jpeg', 'png', 'gif']; // Add more keywords if needed
    const lowercasedMessage = message.toLowerCase();
    return imageKeywords.some(keyword => lowercasedMessage.includes(keyword));
}

// Example usage
const imageUrl = 'https://www.iiitnr.ac.in/sites/default/files/photo_gallary/DSC_0233%20%28Copy%29.jpg';
if (isPhotoLink(imageUrl)) {
    console.log('It is a photo link.');
} else {
    console.log('It is not a photo link.');
}

function isRegularLink(message) {
    // Check if the message is a regular link
    return message.startsWith('http');
}

function createClickableLink(message) {
    // Create a clickable link
    return `<a href="${message}" target="_blank">${message}</a>`;
}

function startSpeechToText() {
    var recognition = new webkitSpeechRecognition() || SpeechRecognition();
    
    recognition.lang = 'en-US';

    recognition.onresult = function (event) {
        var result = event.results[0][0].transcript;
        var messageInput = document.querySelector('.chatbox__footer input');
        messageInput.value = result;

        // Trigger the send button click after getting the transcribed text
        document.querySelector('.chatbox__send--footer').click();
    };

    recognition.start();
}

class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        };

        this.state = false;
        this.message = [];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => {
            console.log("pressed")
            this.toggleState(chatBox)});

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener('keyup', ({ key }) => {
            if (key === 'Enter') {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;

        if (text1 === '') {
            return;
        }

        let msg1 = { name: 'User', message: text1 };
        this.message.push(msg1);

        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: 'Academia', message: r.answer };
            this.message.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';
        })
        .catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
            textField.value = '';
        });
    } 

    // Update the updateChatText function to handle facility photos without creating a new function
// Update the updateChatText function to handle facility photos without creating a new function
// Update the updateChatText function to handle facility photos without creating a new function
// Update the updateChatText function to handle facility photos without creating a new function
updateChatText(chatbox) {
    let html = '';
    const linkRegex = /(https?:\/\/[^\s]+)/g;

    this.message.slice().reverse().forEach(function (item, index) {
        if (item.name === 'Academia') {
            if (isPhotoLink(item.message)) {
                // If it's a link to a photo, create an image element with a description
                const [photoUrl, description] = item.message.split('\n');
                html += `<div class="messages__item messages__item--visitor">
                            <img class="chatbox__image" src="${photoUrl}" alt="Facility Photo">
                            <p>${description}</p>
                        </div>`;
            } else {
                // If it's neither a photo link nor a regular link, display it as a regular message
                const messageWithLinks = item.message.replace(linkRegex, '<a href="$1" target="_blank">$1</a>');
                html += `<div class="messages__item messages__item--visitor">${messageWithLinks}</div>`;
            }
        } else {
            // If it's an operator message, display it as is
            html += `<div class="messages__item messages__item--operator">${item.message}</div>`;
        }
    });

    const chatmessage = chatbox.querySelector('.chatbox__messages');
    chatmessage.innerHTML = html;
}



    
}

const chatbox = new Chatbox();
chatbox.display();